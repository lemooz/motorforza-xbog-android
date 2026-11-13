#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <zlib.h>

#include "kernel/disc.h"
#include "kernel/kernel.h"
#include "kernel/exports.h"

namespace fs = std::filesystem;

namespace kernel {

namespace {

std::mutex g_links_lock;
std::map<std::string, std::string> g_links;    // upper case "\??\D:" -> "\Device\CdRom0"
std::map<std::string, fs::path> g_devices;     // upper case device -> host directory
constexpr char kCdRom[] = "\\DEVICE\\CDROM0";

std::string Upper(std::string s) {
    for (char& c : s) c = char(std::toupper(uint8_t(c)));
    return s;
}

std::string Normalize(std::string p) {
    std::replace(p.begin(), p.end(), '/', '\\');
    std::string out;
    for (char c : p) {
        if (c == '\\' && !out.empty() && out.back() == '\\') continue;
        out += c;
    }
    while (out.size() > 1 && out.back() == '\\') out.pop_back();
    return out;
}

// host lookup that ignores case on file systems that do not
fs::path MatchCase(const fs::path& dir, const std::vector<std::string>& parts) {
    fs::path cur = dir;
    for (auto& part : parts) {
        if (part.empty() || part == ".") continue;
        if (part == "..") {
            cur = cur.parent_path();
            continue;
        }
        fs::path direct = cur / part;
        std::error_code ec;
        if (fs::exists(direct, ec)) {
            cur = direct;
            continue;
        }
        bool found = false;
        std::string want = Upper(part);
        for (auto& e : fs::directory_iterator(cur, ec)) {
            if (Upper(e.path().filename().string()) == want) {
                cur = e.path();
                found = true;
                break;
            }
        }
        if (!found) cur = direct;
    }
    return cur;
}

struct File : Object {
    fs::path host;
    std::string xbox;
    FILE* fp = nullptr;
    const Disc::Entry* entry = nullptr;  // on the disc image instead of a host path
    std::shared_ptr<std::vector<uint8_t>> virtual_data;  // decompressed CA2 exposed as a CAB
    bool virtual_writable = false;       // writable Xbox volume overlay; never modifies the disc image
    bool dir = false;
    uint64_t pos = 0;
    std::vector<fs::directory_entry> entries;
    std::vector<const Disc::Entry*> disc_entries;
    size_t next = 0;
    bool listed = false;
    bool delete_on_close = false;
    std::mutex io;  // threads sharing a handle: seek and transfer stay together
    File() : Object(ObjType::File) {}
    ~File() override {
        if (fp) std::fclose(fp);
        if (delete_on_close) {
            std::error_code ec;
            fs::remove(host, ec);
        }
    }
};

struct Link : Object {
    std::string target;
    Link() : Object(ObjType::SymLink) {}
};

uint64_t ToFileTime(fs::file_time_type t) {
    using namespace std::chrono;
    auto sys = time_point_cast<nanoseconds>(t - fs::file_time_type::clock::now() + system_clock::now());
    return uint64_t(sys.time_since_epoch().count() / 100 + 116444736000000000ll);
}

void FillNetworkInfo(uint32_t out, uint64_t t, uint64_t size, bool dir) {
    for (int i = 0; i < 4; i++) W64(out + i * 8, t);
    W64(out + 0x20, (size + 0xFFF) & ~0xFFFull);
    W64(out + 0x28, size);
    W32(out + 0x30, dir ? 0x10 : 0x80);  // FILE_ATTRIBUTE_DIRECTORY or NORMAL
}

// A CA2 alias can source its bytes from the ISO while keeping an N:/Z:
// namespace. Filesystem queries must describe the guest-visible volume, not
// the host/source entry used to supply the bytes.
bool IsDiscPath(const File& f) {
    return f.xbox.size() >= 2 && std::toupper(uint8_t(f.xbox[0])) == 'D' && f.xbox[1] == ':';
}

void FillNetworkInfo(uint32_t out, const fs::path& p) {
    std::error_code ec;
    bool dir = fs::is_directory(p, ec);
    FillNetworkInfo(out, ToFileTime(fs::last_write_time(p, ec)), dir ? 0 : fs::file_size(p, ec), dir);
}

void FillNetworkInfo(uint32_t out, const Disc::Entry& e) {
    FillNetworkInfo(out, MountedDisc()->Time(), e.dir ? 0 : e.size, e.dir);
}

bool WildMatch(const std::string& pat, const std::string& s) {
    size_t p = 0, i = 0, star = std::string::npos, mark = 0;
    while (i < s.size()) {
        if (p < pat.size() && (pat[p] == '?' || std::toupper(uint8_t(pat[p])) == std::toupper(uint8_t(s[i])))) {
            p++;
            i++;
        } else if (p < pat.size() && pat[p] == '*') {
            star = p++;
            mark = i;
        } else if (star != std::string::npos) {
            p = star + 1;
            i = ++mark;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*') p++;
    return p == pat.size();
}

// past 2 GB a long offset wraps on Windows
void Seek(FILE* fp, uint64_t pos) {
#ifdef _WIN32
    _fseeki64(fp, int64_t(pos), SEEK_SET);
#else
    fseeko(fp, off_t(pos), SEEK_SET);
#endif
}

void IoStatus(uint32_t iosb, uint32_t status, uint32_t info) {
    if (!iosb) return;
    W32(iosb, status);
    W32(iosb + 4, info);
}

std::shared_ptr<File> FileFromHandle(uint32_t h) {
    ObjectRef o = FromHandle(h);
    if (!o || o->type != ObjType::File) return nullptr;
    return std::static_pointer_cast<File>(o);
}

std::string ObjectName(uint32_t attrs, uint32_t* root = nullptr) {
    if (root) *root = R32(attrs);
    return ReadAnsiString(R32(attrs + 4));
}

void Complete(uint32_t event, uint32_t apc, uint32_t apc_ctx, uint32_t iosb) {
    static std::atomic<uint32_t> apc_completions{0};
    if (apc) {
        const uint32_t n = apc_completions.fetch_add(1, std::memory_order_relaxed);
        if (n < 64)
            LOGR("diag: io-apc-complete[%u] event=%08X routine=%08X context=%08X iosb=%08X status=%08X info=%u thread=%X",
                 n, event, apc, apc_ctx, iosb, iosb ? R32(iosb) : 0, iosb ? R32(iosb + 4) : 0,
                 Current() ? Current()->id : 0);
    }
    if (event)
        if (ObjectRef e = FromHandle(event)) SetEvent(*e);
    if (apc && Current()) QueueApc(*Current(), apc, apc_ctx, iosb, 0);
}

}  // namespace

std::string ReadAnsiString(uint32_t str) {
    if (!str) return {};
    uint16_t len = R16(str);
    uint32_t buf = R32(str + 4);
    if (!buf) return {};
    return std::string(reinterpret_cast<const char*>(HostPtr(buf)), len);
}

std::string ReadCString(uint32_t va, size_t max) {
    std::string s;
    if (!va) return s;
    for (size_t i = 0; i < max; i++) {
        char c = char(G<uint8_t>(va + uint32_t(i)));
        if (!c) break;
        s += c;
    }
    return s;
}

void MountDefaults(const std::string& game_dir, const std::string& data_dir) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    fs::path d = data_dir;
    g_devices[kCdRom] = game_dir;
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION1"] = d / "E";
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION2"] = d / "C";
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION3"] = d / "X";
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION4"] = d / "Y";
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION5"] = d / "Z";
    // Forza probes N:\ as a directory during title initialization and stages
    // its CA2-backed CABs there. Keep the volume writable and separate from
    // the read-only disc image, just like the other emulated hard-disk volumes.
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION6"] = d / "N";
    for (auto& [k, v] : g_devices) {
        std::error_code ec;
        fs::create_directories(v, ec);
    }
    g_links["\\??\\D:"] = "\\Device\\CdRom0";
    g_links["\\??\\N:"] = "\\Device\\Harddisk0\\Partition6";
    const Disc* disc = MountedDisc();
    LOGI("files: D: is %s, hard disk under %s", disc ? disc->Path().string().c_str() : game_dir.c_str(), data_dir.c_str());
}

bool CreateSymbolicLink(const std::string& link, const std::string& target) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    g_links[Upper(Normalize(link))] = Normalize(target);
    LOGI("files: link %s -> %s", link.c_str(), target.c_str());
    return true;
}

bool DeleteSymbolicLink(const std::string& link) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    return g_links.erase(Upper(Normalize(link))) != 0;
}

std::string QuerySymbolicLink(const std::string& link) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    auto it = g_links.find(Upper(Normalize(link)));
    return it == g_links.end() ? std::string() : it->second;
}

namespace {

struct Target {
    std::string host;                    // empty on the disc image or unmapped
    bool disc = false;                   // on the mounted disc image
    const Disc::Entry* entry = nullptr;  // there: the file or folder, null when missing
    bool ca2 = false;                    // the retail media stores the requested CAB as block-zlib CA2
};

Target Resolve(const std::string& xbox_path, uint32_t root_handle) {
    std::string p = Normalize(xbox_path);
    if (root_handle) {
        if (auto f = FileFromHandle(root_handle)) p = Normalize(f->xbox + "\\" + p);
        // The title uses its XBE CD-media root pseudo-handle for relative
        // opens (for example, "CarIcons1.xpr"). It is not a kernel File
        // handle, so resolve those names against the media directory.
        else if (root_handle == 0xFFFFFFFDu && p.find(':') == std::string::npos) {
            // XAPI passes both bare names and root-relative names beginning
            // with '\\' for the XBE CD-media handle. Neither is an NT path.
            while (!p.empty() && p[0] == '\\') p.erase(p.begin());
            p = Normalize("D:\\MEDIA\\" + p);
        }
    }
    // Forza's asset loader requests CAB bundles from N: and Z:, while this
    // retail disc stores the compressed containers under D:\\Media\\cabs as CA2.
    // Resolve those virtual names to the read-only XDVDFS entries.
    const std::string requested = Upper(p);
    const bool ca2 = requested == "N:\\CARS.CAB" || requested == "N:\\MEGA.CAB" ||
                     requested == "Z:\\MEGAXPR.CAB" || requested == "Z:\\UI_NTSC.CAB";
    if (requested == "N:\\CARS.CAB") p = "D:\\Media\\cabs\\cars.ca2";
    else if (requested == "N:\\MEGA.CAB") p = "D:\\Media\\cabs\\mega.ca2";
    else if (requested == "Z:\\MEGAXPR.CAB") p = "D:\\Media\\cabs\\megaxpr.ca2";
    else if (requested == "Z:\\UI_NTSC.CAB") p = "D:\\Media\\cabs\\ui_ntsc.ca2";
    if (p.size() >= 2 && p[1] == ':') p = "\\??\\" + p;
    std::lock_guard<std::mutex> lock(g_links_lock);
    for (int depth = 0; depth < 8; depth++) {
        std::string up = Upper(p);
        bool changed = false;
        for (auto& [k, v] : g_links) {
            if (up.compare(0, k.size(), k) == 0 && (up.size() == k.size() || up[k.size()] == '\\')) {
                p = v + p.substr(k.size());
                changed = true;
                break;
            }
        }
        if (!changed) break;
    }
    std::string up = Upper(p);
    const std::pair<const std::string, fs::path>* best = nullptr;
    for (auto& e : g_devices)
        if (up.compare(0, e.first.size(), e.first) == 0 && (up.size() == e.first.size() || up[e.first.size()] == '\\'))
            if (!best || e.first.size() > best->first.size()) best = &e;
    if (!best) return {};
    std::vector<std::string> parts;
    std::string rest = p.substr(best->first.size());
    size_t s = 0;
    while (s < rest.size()) {
        size_t e = rest.find('\\', s);
        if (e == std::string::npos) e = rest.size();
        if (e > s) parts.push_back(rest.substr(s, e - s));
        s = e + 1;
    }
    // no Xbox name holds ':', and fs::path would take "D:" as a host drive
    for (auto& part : parts)
        if (part.find(':') != std::string::npos) return {};
    if (best->first == kCdRom)
        if (const Disc* disc = MountedDisc()) return {{}, true, disc->Find(parts), ca2};
    return {MatchCase(best->second, parts).string()};
}

// the disc image is read only: nothing on it is created, replaced or written
std::shared_ptr<std::vector<uint8_t>> DecompressCa2(const Disc::Entry& entry) {
    static std::mutex cache_lock;
    static std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint8_t>>> cache;
    std::lock_guard<std::mutex> lock(cache_lock);
    if (auto it = cache.find(entry.offset); it != cache.end()) return it->second;

    std::vector<uint8_t> compressed(entry.size);
    if (MountedDisc()->Read(entry, 0, compressed.data(), entry.size) != entry.size) return {};
    auto decoded = std::make_shared<std::vector<uint8_t>>();
    size_t pos = 0;
    while (pos + 2 <= compressed.size()) {
        // CA2 concatenates independent zlib streams, each beginning on a 2 KiB boundary.
        z_stream stream{};
        if (inflateInit(&stream) != Z_OK) return {};
        stream.next_in = compressed.data() + pos;
        stream.avail_in = static_cast<uInt>(std::min<size_t>(compressed.size() - pos, std::numeric_limits<uInt>::max()));
        std::array<uint8_t, 64 * 1024> chunk{};
        int result = Z_OK;
        const size_t before = decoded->size();
        while (result == Z_OK) {
            stream.next_out = chunk.data();
            stream.avail_out = static_cast<uInt>(chunk.size());
            result = inflate(&stream, Z_NO_FLUSH);
            const size_t produced = chunk.size() - stream.avail_out;
            decoded->insert(decoded->end(), chunk.begin(), chunk.begin() + produced);
            if (result == Z_OK && stream.avail_in == 0 && stream.avail_out != 0) break;
        }
        const size_t consumed = stream.total_in;
        inflateEnd(&stream);
        if (result != Z_STREAM_END || consumed == 0) {
            LOGR("diag: ca2-inflate-failed path=%s offset=%zu zlib=%d input=%zu output=%zu",
                 entry.name.c_str(), pos, result, compressed.size(), decoded->size() - before);
            return {};
        }
        pos = (pos + consumed + 2047) & ~size_t(2047);
    }
    if (decoded->empty()) return {};
    LOGR("diag: ca2-decoded path=%s compressed=%u uncompressed=%zu", entry.name.c_str(), entry.size, decoded->size());
    cache.emplace(entry.offset, decoded);
    return decoded;
}

uint32_t OpenOnDisc(const Disc::Entry* e, const std::string& name, uint32_t handle_out, uint32_t iosb, uint32_t disposition,
                    uint32_t options, bool ca2, uint32_t access) {
    static std::atomic<uint32_t> opens{0}, failures{0};
    // Some XAPI asset probes use OVERWRITE_IF while asking only for read
    // access. On the read-only XDVDFS image that is a read-open, not a request
    // to mutate the disc. Preserve write failures while allowing that probe.
    constexpr uint32_t kWriteAccess = 0x40000000u | 0x10000000u | 0x000D0000u | 0x00000116u;
    if (!e && disposition == 5) {
        // The title creates its generated CarIcons.sig beside assets on the
        // nominal CD volume. Keep that generated file in RAM instead.
        auto f = std::make_shared<File>();
        f->xbox = Normalize(name);
        f->virtual_writable = true;
        f->virtual_data = std::make_shared<std::vector<uint8_t>>();
        W32(handle_out, NewHandle(f));
        IoStatus(iosb, kSuccess, 2);
        LOGR("diag: disc-overlay-create path=%s access=%08X", name.c_str(), access);
        return kSuccess;
    }
    if (e && disposition == 5 && !(access & kWriteAccess)) disposition = 1;
    if (e && disposition == 5) {
        // XAPI occasionally asks to OVERWRITE_IF a file that lives on the
        // read-only CD image. Give it a RAM copy-on-write handle so the read
        // and any later writes proceed without modifying the mounted image.
        auto f = std::make_shared<File>();
        f->xbox = Normalize(name);
        f->entry = e;
        f->virtual_writable = true;
        f->virtual_data = std::make_shared<std::vector<uint8_t>>(e->size);
        const uint32_t got = MountedDisc()->Read(*e, 0, f->virtual_data->data(), e->size);
        if (got != e->size) {
            IoStatus(iosb, kUnsuccessful, 0);
            return kUnsuccessful;
        }
        f->dir = e->dir;
        W32(handle_out, NewHandle(f));
        IoStatus(iosb, kSuccess, 1);
        LOGR("diag: disc-overlay-open path=%s size=%u access=%08X", name.c_str(), e->size, access);
        return kSuccess;
    }
    if (!e && (disposition == 1 || disposition == 4)) {  // FILE_OPEN or FILE_OVERWRITE need an existing file
        LOG_FIRST(200, "missing %s", name.c_str());
        uint32_t n = failures.fetch_add(1, std::memory_order_relaxed);
        if (n < 64) LOGR("diag: disc-open-fail[%u] path=%s disposition=%u options=%08X status=%08X",
                         n, name.c_str(), disposition, options, kObjectNameNotFound);
        IoStatus(iosb, kObjectNameNotFound, 5);
        return kObjectNameNotFound;
    }
    if (!e || (disposition != 1 && disposition != 3)) {
        uint32_t n = failures.fetch_add(1, std::memory_order_relaxed);
        if (n < 64) LOGR("diag: disc-open-fail[%u] path=%s disposition=%u options=%08X status=%08X",
                         n, name.c_str(), disposition, options, kAccessDenied);
        IoStatus(iosb, kAccessDenied, 0);
        return kAccessDenied;
    }
    if (e->dir && !(options & 1) && (options & 0x40)) {  // options 1 FILE_DIRECTORY_FILE, 0x40 FILE_NON_DIRECTORY_FILE
        uint32_t n = failures.fetch_add(1, std::memory_order_relaxed);
        if (n < 64) LOGR("diag: disc-open-fail[%u] path=%s disposition=%u options=%08X status=%08X directory=1",
                         n, name.c_str(), disposition, options, kFileIsADirectory);
        IoStatus(iosb, kFileIsADirectory, 0);
        return kFileIsADirectory;
    }
    auto f = std::make_shared<File>();
    f->xbox = Normalize(name);
    f->entry = e;
    if (ca2) {
        auto decoded = DecompressCa2(*e);
        if (!decoded) {
            IoStatus(iosb, kUnsuccessful, 0);
            return kUnsuccessful;
        }
        // N:/Z: are guest hard-disk volumes. The title stages CAB entries by
        // copying them to a writable handle with the same virtual name. Keep
        // that handle's mutations in RAM; other read handles keep the ISO view.
        f->virtual_writable = (access & 0x40000000u) != 0;
        f->virtual_data = f->virtual_writable
                              ? std::make_shared<std::vector<uint8_t>>(*decoded)
                              : std::move(decoded);
        static std::atomic<uint32_t> ca2_open_logs{0};
        if (ca2_open_logs.fetch_add(1, std::memory_order_relaxed) < 48)
            LOGR("diag: ca2-open path=%s access=%08X writable=%u logical-size=%zu",
                 name.c_str(), access, f->virtual_writable, f->virtual_data->size());
    }
    f->dir = e->dir;
    W32(handle_out, NewHandle(f));
    IoStatus(iosb, kSuccess, 1);
    LOG_FIRST(400, "open %s", name.c_str());
    uint32_t n = opens.fetch_add(1, std::memory_order_relaxed);
    if (n < 96) LOGR("diag: disc-open[%u] path=%s size=%u dir=%u offset=%llu",
                     n, name.c_str(), e->size, e->dir, (unsigned long long)e->offset);
    return kSuccess;
}

}  // namespace

std::string ResolvePath(const std::string& xbox_path, uint32_t root_handle) { return Resolve(xbox_path, root_handle).host; }

bool ReadGameFile(const std::string& xbox_path, std::vector<uint8_t>& out) {
    Target t = Resolve(xbox_path, 0);
    if (t.disc) {
        if (!t.entry || t.entry->dir) {
            LOGR("diag: game-file-missing path=%s", xbox_path.c_str());
            return false;
        }
        if (t.ca2) {
            auto decoded = DecompressCa2(*t.entry);
            if (!decoded) return false;
            out = *decoded;
            return true;
        }
        out.resize(t.entry->size);
        uint32_t got = MountedDisc()->Read(*t.entry, 0, out.data(), t.entry->size);
        if (got != t.entry->size)
            LOGR("diag: game-file-short-read path=%s got=%u size=%u", xbox_path.c_str(), got, t.entry->size);
        return got == t.entry->size;
    }
    std::ifstream f(t.host, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

// exports

KERNEL_EXPORT(190, NtCreateFile, 9) {
    uint32_t handle_out = a[0], access = a[1], attrs = a[2], iosb = a[3], disposition = a[7], options = a[8];
    (void)access;
    uint32_t root;
    std::string name = ObjectName(attrs, &root);
    const std::string upper_name = Upper(name);
    const bool save_meta = upper_name.find("\\UDATA\\") != std::string::npos &&
                           (upper_name.ends_with("\\SAVEMETA.XBX") || upper_name.ends_with("\\SAVEIMAGE.XBX"));
    static std::atomic<uint32_t> save_meta_open_logs{0};
    if (save_meta && save_meta_open_logs.fetch_add(1, std::memory_order_relaxed) < 128)
        LOGR("diag: save-file-open path=%s access=%08X root=%08X disposition=%u options=%08X",
             name.c_str(), access, root, disposition, options);
    Target t = Resolve(name, root);
    if (t.disc) return OpenOnDisc(t.entry, name, handle_out, iosb, disposition, options, t.ca2, access);
    const std::string& host = t.host;
    if (host.empty()) {
        LOGR("diag: NtCreateFile unmapped xbox=%s root=%08X disposition=%u options=%08X", name.c_str(), root, disposition, options);
        IoStatus(iosb, kObjectPathNotFound, 0);
        return kObjectPathNotFound;
    }
    std::error_code ec;
    bool exists = fs::exists(host, ec);
    bool is_dir = exists && fs::is_directory(host, ec);
    bool want_dir = options & 1;
    uint32_t info = 1;  // FILE_OPENED
    if (want_dir || is_dir) {
        if (!exists) {
            if (disposition == 1 || disposition == 4) {
                IoStatus(iosb, kObjectNameNotFound, 5);
                return kObjectNameNotFound;
            }
            fs::create_directories(host, ec);
            info = 2;
        } else if (disposition == 2) {
            // Opening a first-run SaveMeta.xbx above can materialize the profile's
            // parent directory (the Xbox save container is implicit). The title
            // then issues FILE_CREATE for that same directory. Treat only this
            // empty, just-bootstrapped profile container as already opened; keep
            // normal FILE_CREATE collision semantics for every other directory.
            bool bootstrap_profile_dir = false;
            const fs::path profile_path(host);
            const fs::path profile_parent = profile_path.parent_path();
            const fs::path save_title_root = profile_parent.parent_path();
            if ((name.size() >= 2 && (name[0] == 'U' || name[0] == 'u' || name[0] == 'T' || name[0] == 't')) &&
                Upper(profile_parent.filename().string()) == "4D53006E" &&
                Upper(save_title_root.filename().string()) == "UDATA" &&
                profile_path.filename().string().size() == 12) {
                    const fs::path meta = fs::path(host) / "SaveMeta.xbx";
                    ec.clear();
                    if (fs::is_regular_file(meta, ec) && !ec && fs::file_size(meta, ec) == 0 && !ec) {
                        fs::directory_iterator it(host, ec), end;
                        bootstrap_profile_dir = !ec && it != end && it->path().filename() == "SaveMeta.xbx";
                        if (bootstrap_profile_dir) {
                            ++it;
                            bootstrap_profile_dir = !ec && it == end;
                        }
                    }
            }
            if (bootstrap_profile_dir) {
                LOGR("diag: NtCreateFile-opened-bootstrap-profile-dir xbox=%s host=%s", name.c_str(), host.c_str());
            } else {
                static std::atomic<uint32_t> collision_logs{0};
                if (collision_logs.fetch_add(1, std::memory_order_relaxed) < 64)
                    LOGR("diag: NtCreateFile-collision xbox=%s host=%s disposition=%u options=%08X directory=1",
                         name.c_str(), host.c_str(), disposition, options);
                IoStatus(iosb, kObjectNameCollision, 4);
                return kObjectNameCollision;
            }
        }
        if (!want_dir && (options & 0x40)) {
            IoStatus(iosb, kFileIsADirectory, 0);
            return kFileIsADirectory;
        }
        auto f = std::make_shared<File>();
        f->host = host;
        f->xbox = Normalize(name);
        f->dir = true;
        W32(handle_out, NewHandle(f));
        IoStatus(iosb, kSuccess, info);
        return kSuccess;
    }
    const char* mode = nullptr;
    switch (disposition) {
        case 0:  // supersede
        case 5:  // overwrite if
            mode = "w+b";
            info = exists ? 3 : 2;
            break;
        case 1:  // open
            if (!exists) {
                LOG_FIRST(200, "missing %s", name.c_str());
                IoStatus(iosb, kObjectNameNotFound, 5);
                return kObjectNameNotFound;
            }
            mode = "r+b";
            break;
        case 2:  // create
            if (exists) {
                static std::atomic<uint32_t> collision_logs{0};
                if (collision_logs.fetch_add(1, std::memory_order_relaxed) < 64)
                    LOGR("diag: NtCreateFile-collision xbox=%s host=%s disposition=%u options=%08X directory=0",
                         name.c_str(), host.c_str(), disposition, options);
                IoStatus(iosb, kObjectNameCollision, 4);
                return kObjectNameCollision;
            }
            mode = "w+b";
            info = 2;
            break;
        case 3:  // open if
            mode = exists ? "r+b" : "w+b";
            info = exists ? 1 : 2;
            break;
        case 4:  // overwrite
            if (!exists) {
                IoStatus(iosb, kObjectNameNotFound, 5);
                return kObjectNameNotFound;
            }
            mode = "w+b";
            info = 3;
            break;
        default:
            return kInvalidParameter;
    }
    // XAPI opens save metadata with OPEN_IF before the per-profile directory exists.
    // Ensure the title's save containers are present so that first-run profile setup can proceed.
    if (!exists && !fs::is_directory(fs::path(host).parent_path(), ec)) {
        const bool save_volume = name.size() >= 2 && name[1] == ':' &&
                                 (name[0] == 'U' || name[0] == 'u' || name[0] == 'T' || name[0] == 't');
        if (save_volume && disposition == 3) {
            ec.clear();
            fs::create_directories(fs::path(host).parent_path(), ec);
            if (!ec)
                LOGR("diag: NtCreateFile save-parent-created xbox=%s host=%s", name.c_str(), host.c_str());
        }
        if (ec || !fs::is_directory(fs::path(host).parent_path(), ec)) {
            LOGR("diag: NtCreateFile missing-parent xbox=%s host=%s root=%08X disposition=%u options=%08X", name.c_str(), host.c_str(), root, disposition, options);
            IoStatus(iosb, kObjectPathNotFound, 0);
            return kObjectPathNotFound;
        }
    }
    FILE* fp = std::fopen(host.c_str(), mode);
    if (!fp && std::string(mode) == "r+b") fp = std::fopen(host.c_str(), "rb");  // read only media
    if (!fp) {
        LOGW("NtCreateFile: cannot open %s", host.c_str());
        IoStatus(iosb, kAccessDenied, 0);
        return kAccessDenied;
    }
    auto f = std::make_shared<File>();
    f->host = host;
    f->xbox = Normalize(name);
    f->fp = fp;
    f->delete_on_close = options & 0x1000;  // FILE_DELETE_ON_CLOSE
    W32(handle_out, NewHandle(f));
    IoStatus(iosb, kSuccess, info);
    if (save_meta) {
        const uint64_t size = fs::file_size(host, ec);
        LOGR("diag: save-file-opened path=%s host=%s handle=%08X info=%u size=%llu mode=%s",
             name.c_str(), host.c_str(), R32(handle_out), info, (unsigned long long)size, mode);
    }
    LOG_FIRST(400, "open %s", name.c_str());
    return kSuccess;
}

KERNEL_EXPORT(202, NtOpenFile, 6) {
    // same as create with FILE_OPEN
    uint32_t handle_out = a[0], access = a[1], attrs = a[2], iosb = a[3], options = a[5];
    uint32_t root;
    std::string name = ObjectName(attrs, &root);
    Target t = Resolve(name, root);
    if (t.disc) return OpenOnDisc(t.entry, name, handle_out, iosb, 1, options, t.ca2, access);
    const std::string& host = t.host;
    std::error_code ec;
    if (host.empty() || !fs::exists(host, ec)) {
        LOG_FIRST(200, "missing %s", name.c_str());
        static std::atomic<uint32_t> missing_open_logs{0};
        uint32_t missing_n = missing_open_logs.fetch_add(1, std::memory_order_relaxed);
        if (missing_n < 64)
            LOGR("diag: NtOpenFile-missing[%u] caller=%08X path=%s host=%s root=%08X options=%08X",
                 missing_n, R32(a.ctx.esp), name.c_str(), host.c_str(), root, options);
        IoStatus(iosb, kObjectNameNotFound, 5);
        return host.empty() ? kObjectPathNotFound : kObjectNameNotFound;
    }
    auto f = std::make_shared<File>();
    f->host = host;
    f->xbox = Normalize(name);
    f->dir = fs::is_directory(host, ec);
    if (!f->dir) {
        f->fp = std::fopen(host.c_str(), "r+b");
        if (!f->fp) f->fp = std::fopen(host.c_str(), "rb");
        if (!f->fp) return kAccessDenied;
    }
    (void)options;
    W32(handle_out, NewHandle(f));
    IoStatus(iosb, kSuccess, 1);
    return kSuccess;
}

// I/O completes before the call returns: the status block is written, then the file object, the event and the
// APC are signalled (GetOverlappedResult without an event waits on the file handle)
static uint32_t Finish(File& f, uint32_t status, uint32_t n, uint32_t event, uint32_t apc, uint32_t apc_ctx, uint32_t iosb) {
    IoStatus(iosb, status, n);
    SetEvent(f);
    Complete(event, apc, apc_ctx, iosb);
    return status;
}

static uint64_t Offset(File& f, uint32_t off) {
    int64_t o = off ? int64_t(R64(off)) : -1;
    return o >= 0 ? uint64_t(o) : f.pos;
}

// pages is the FILE_SEGMENT_ELEMENT array for scatter and gather, 0 for one buffer
static uint32_t Transfer(const Args& a, bool write, uint32_t pages) {
    static std::atomic<uint32_t> disc_reads{0}, disc_short_reads{0};
    static std::mutex read_log_lock;
    static std::unordered_map<std::string, uint32_t> reads_by_file;
    uint32_t h = a[0], event = a[1], apc = a[2], apc_ctx = a[3], iosb = a[4], buf = a[5], len = a[6], off = a[7];
    auto f = FileFromHandle(h);
    if (!f || (!f->fp && !f->entry && !f->virtual_data)) return kInvalidHandle;
    const std::string upper_path = Upper(f->xbox);
    const bool save_meta = upper_path.find("\\UDATA\\") != std::string::npos &&
                           (upper_path.ends_with("\\SAVEMETA.XBX") || upper_path.ends_with("\\SAVEIMAGE.XBX"));
    static std::atomic<uint32_t> save_meta_io_logs{0};
    if (save_meta && save_meta_io_logs.fetch_add(1, std::memory_order_relaxed) < 128)
        LOGR("diag: save-file-io op=%s path=%s handle=%08X pos=%llu len=%u buffer=%08X",
             write ? "write" : "read", f->xbox.c_str(), h,
             (unsigned long long)Offset(*f, off), len, buf);
    if (write) {
        static std::atomic<uint32_t> write_logs{0};
        const uint32_t nlog = write_logs.fetch_add(1, std::memory_order_relaxed);
        if (nlog < 64)
            LOGR("diag: file-write[%u] handle=%08X path=%s entry=%u ca2=%u host=%s pos=%llu len=%u",
                 nlog, h, f->xbox.c_str(), f->entry != nullptr, f->virtual_data != nullptr,
                 f->host.c_str(), (unsigned long long)Offset(*f, off), len);
    }
    if (write && f->entry && !f->virtual_writable) return kAccessDenied;
    ResetEvent(*f);
    std::lock_guard<std::mutex> lock(f->io);
    uint64_t pos = Offset(*f, off);
    if (f->fp) Seek(f->fp, pos);
    uint32_t n = 0;
    while (n < len) {
        uint32_t at = pages ? R32(pages + n / 0x1000 * 8) + n % 0x1000 : buf + n;
        uint32_t chunk = pages ? std::min(len - n, 0x1000 - n % 0x1000) : len;
        size_t done = write && f->virtual_writable
                      ? size_t(chunk)
                      : f->virtual_data
                      ? (pos + n < f->virtual_data->size()
                             ? std::min<size_t>(chunk, f->virtual_data->size() - size_t(pos + n))
                             : 0)
                      : f->entry ? MountedDisc()->Read(*f->entry, pos + n, HostPtr(at), chunk)
                      : write  ? std::fwrite(HostPtr(at), 1, chunk, f->fp)
                               : std::fread(HostPtr(at), 1, chunk, f->fp);
        if (write && f->virtual_writable && done) {
            const size_t end = size_t(pos + n) + done;
            if (end > f->virtual_data->size()) f->virtual_data->resize(end);
            std::memcpy(f->virtual_data->data() + pos + n, HostPtr(at), done);
        } else if (f->virtual_data && done) {
            std::memcpy(HostPtr(at), f->virtual_data->data() + pos + n, done);
        }
        n += uint32_t(done);
        if (done < chunk) break;
    }
    f->pos = pos + n;
    if (save_meta)
        LOGR("diag: save-file-io-complete op=%s path=%s pos=%llu request=%u transferred=%u status=%08X",
             write ? "write" : "read", f->xbox.c_str(), (unsigned long long)pos, len, n,
             write && n < len ? 0xC000007Fu : (!write && len && !n && !f->entry ? kEndOfFile : kSuccess));
    if (f->entry && !write) {
        uint32_t read_n = disc_reads.fetch_add(1, std::memory_order_relaxed);
        uint32_t file_read_n;
        {
            std::lock_guard<std::mutex> log_lock(read_log_lock);
            file_read_n = reads_by_file[f->xbox]++;
        }
        if (pos == 0 || file_read_n < 48) {
            uint32_t frame0 = a.ctx.ebp;
            LOGR("diag: disc-read[%u] caller=%08X ebp=%08X prev=%08X ret=%08X path=%s pos=%llu request=%u got=%u size=%u",
                 read_n, R32(a.ctx.esp), frame0, R32(frame0), R32(frame0 + 4),
                 f->xbox.c_str(), (unsigned long long)pos, len, n, f->entry->size);
        }
        if (f->virtual_data && pos == 0 && n >= 16)
            LOGR("diag: ca2-cab-header path=%s bytes=%08X %08X %08X %08X logical-size=%zu",
                 f->xbox.c_str(), R32(buf), R32(buf + 4), R32(buf + 8), R32(buf + 12), f->virtual_data->size());
        if (n < len) {
            uint32_t short_n = disc_short_reads.fetch_add(1, std::memory_order_relaxed);
            if (short_n < 64) LOGR("diag: disc-short-read[%u] caller=%08X path=%s pos=%llu request=%u got=%u size=%u",
                                   short_n, R32(a.ctx.esp), f->xbox.c_str(), (unsigned long long)pos,
                                   len, n, f->entry->size);
        }
    }
    if (write && n < len) return Finish(*f, 0xC000007F, n, event, apc, apc_ctx, iosb);  // disk full
    // XDVDFS async streams use a speculative second 128 KiB read even when the
    // first read reached the end of a short file. The Xbox title treats that
    // EOF completion as a media failure; keep the zero-byte completion
    // successful for disc files so the stream consumer can observe EOF through
    // the transferred-byte count (the partial final read is already success).
    if (!write && len && !n) {
        if (f->entry) return Finish(*f, kSuccess, 0, event, apc, apc_ctx, iosb);
        return Finish(*f, kEndOfFile, 0, event, apc, apc_ctx, iosb);
    }
    return Finish(*f, kSuccess, n, event, apc, apc_ctx, iosb);
}

KERNEL_EXPORT(219, NtReadFile, 8) { return Transfer(a, false, 0); }
KERNEL_EXPORT(236, NtWriteFile, 8) { return Transfer(a, true, 0); }
// the buffer argument is an array of one page per element
KERNEL_EXPORT(220, NtReadFileScatter, 8) { return Transfer(a, false, a[5]); }
KERNEL_EXPORT(237, NtWriteFileGather, 8) { return Transfer(a, true, a[5]); }

KERNEL_EXPORT(211, NtQueryInformationFile, 5) {
    uint32_t h = a[0], iosb = a[1], out = a[2], len = a[3], cls = a[4];
    auto f = FileFromHandle(h);
    if (!f) return kInvalidHandle;
    std::error_code ec;
    uint64_t size = f->dir ? 0 : f->virtual_data ? f->virtual_data->size()
                            : f->entry ? f->entry->size : fs::file_size(f->host, ec);
    if (f->fp) {
        std::fflush(f->fp);
        size = fs::file_size(f->host, ec);
    }
    static std::atomic<uint32_t> file_standard_queries{0};
    std::memset(HostPtr(out), 0, len);
    switch (cls) {
        case 4: {  // FileBasicInformation
            uint64_t t = f->entry ? MountedDisc()->Time() : ToFileTime(fs::last_write_time(f->host, ec));
            for (int i = 0; i < 4; i++) W64(out + i * 8, t);
            W32(out + 0x20, f->dir ? 0x10 : 0x80);
            IoStatus(iosb, kSuccess, 0x28);
            return kSuccess;
        }
        case 5: {  // FileStandardInformation
            // XDVDFS allocates files in 2048-byte sectors; host/HDD files use
            // the runtime's 4096-byte allocation unit.
            const bool dvd = IsDiscPath(*f);
            W64(out, (size + (dvd ? 0x7FF : 0xFFF)) & (dvd ? ~uint64_t(0x7FF) : ~uint64_t(0xFFF)));
            W64(out + 8, size);
            W32(out + 0x10, 1);  // NumberOfLinks
            G<uint8_t>(out + 0x15) = f->dir;  // Directory, after DeletePending
            if (f->entry && file_standard_queries.fetch_add(1, std::memory_order_relaxed) < 64)
                LOGR("diag: file-standard path=%s size=%llu allocation=%llu",
                     f->xbox.c_str(), (unsigned long long)size, (unsigned long long)R64(out));
            IoStatus(iosb, kSuccess, 0x18);
            return kSuccess;
        }
        case 14:  // FilePositionInformation
            W64(out, f->pos);
            IoStatus(iosb, kSuccess, 8);
            return kSuccess;
        case 34:  // FileNetworkOpenInformation
            if (f->entry) {
                const uint64_t logical_size = f->virtual_data ? f->virtual_data->size() : f->entry->size;
                FillNetworkInfo(out, MountedDisc()->Time(), f->dir ? 0 : logical_size, f->dir);
                static std::atomic<uint32_t> logical_queries{0};
                if (f->virtual_data && logical_queries.fetch_add(1, std::memory_order_relaxed) < 32)
                    LOGR("diag: file-network-info path=%s logical-size=%llu compressed-size=%u",
                         f->xbox.c_str(), (unsigned long long)logical_size, f->entry->size);
            }
            else FillNetworkInfo(out, f->host);
            IoStatus(iosb, kSuccess, 0x38);
            return kSuccess;
        case 22:  // FileModeInformation
        case 16:
            IoStatus(iosb, kSuccess, 4);
            return kSuccess;
        default:
            LOG_FIRST(8, "NtQueryInformationFile: class %u", cls);
            return kInvalidParameter;
    }
}

KERNEL_EXPORT(226, NtSetInformationFile, 5) {
    uint32_t h = a[0], iosb = a[1], in = a[2], cls = a[4];
    auto f = FileFromHandle(h);
    if (!f) return kInvalidHandle;
    static std::atomic<uint32_t> set_info_logs{0};
    if (set_info_logs.fetch_add(1, std::memory_order_relaxed) < 64)
        LOGR("diag: file-set-info handle=%08X path=%s class=%u ca2=%u writable=%u value=%llu",
             h, f->xbox.c_str(), cls, f->entry != nullptr, f->virtual_writable,
             (unsigned long long)R64(in));
    // CA2-backed CAB handles are an in-memory writable staging copy. The
    // title sets allocation/end-of-file information after copying the stream;
    // treating these like the read-only source entry leaves the staged CAB
    // with the wrong logical length.
    if (f->entry && !f->virtual_writable && cls != 14 && cls != 4) return kAccessDenied;
    std::error_code ec;
    switch (cls) {
        case 14:  // position
            f->pos = R64(in);
            break;
        case 19:  // allocation: only a size below the end of file cuts it
        case 20:  // end of file
            if (f->virtual_writable && f->virtual_data) {
                const uint64_t requested = R64(in);
                if (requested > std::numeric_limits<size_t>::max()) return kInvalidParameter;
                if (cls == 20 || requested < f->virtual_data->size())
                    f->virtual_data->resize(size_t(requested));
                else if (requested > f->virtual_data->capacity())
                    f->virtual_data->reserve(size_t(requested));
            } else if (f->fp) {
                std::lock_guard<std::mutex> lock(f->io);
                std::fflush(f->fp);
                uint64_t size = fs::file_size(f->host, ec);
                if (cls == 20 || R64(in) < size) fs::resize_file(f->host, R64(in), ec);
            }
            break;
        case 13:  // disposition
            f->delete_on_close = G<uint8_t>(in) != 0;
            break;
        case 10: {  // rename: ReplaceIfExists, RootDirectory, FileName (ANSI_STRING inline)
            uint32_t root = R32(in + 4);
            std::string target = ReadAnsiString(in + 8);
            std::string host = ResolvePath(target, root);
            if (host.empty()) return kObjectPathNotFound;
            if (f->fp) std::fflush(f->fp);
            fs::rename(f->host, host, ec);
            if (ec) return kAccessDenied;
            f->host = host;
            break;
        }
        case 4:  // basic, times and attributes are not kept
            break;
        default:
            LOG_FIRST(8, "NtSetInformationFile: class %u", cls);
            return kInvalidParameter;
    }
    IoStatus(iosb, kSuccess, 0);
    return kSuccess;
}

// Shared FileFs*Information fill for NtQueryVolumeInformationFile and IoQueryVolumeInformation
static uint32_t QueryVolumeClass(uint32_t handle, uint32_t out, uint32_t len, uint32_t cls, uint32_t iosb,
                                 const char* who, bool io_return_length = false) {
    static std::atomic<uint32_t> volume_queries{0};
    uint32_t n = volume_queries.fetch_add(1, std::memory_order_relaxed);
    ObjectRef volume = FromHandle(handle);
    if (!volume) volume = FromGuest(handle);
    auto f = volume && volume->type == ObjType::File ? std::static_pointer_cast<File>(volume) : nullptr;
    const bool dvd = f && IsDiscPath(*f);
    auto completed = [&](uint32_t bytes) {
        if (io_return_length) {
            if (iosb) W32(iosb, bytes);
        } else {
            IoStatus(iosb, kSuccess, bytes);
        }
    };
    if (n < 64) LOGR("diag: volume-query[%u] via=%s class=%u length=%u handle=%08X media=%s path=%s",
                     n, who, cls, len, handle, dvd ? "XDVDFS" : "FATX", f ? f->xbox.c_str() : "<device>");
    std::memset(HostPtr(out), 0, len);
    switch (cls) {
        case 3:  // FileFsSizeInformation: 16 GiB total, 8 GiB free in 16 KiB units
            if (dvd) {
                W64(out, (MountedDisc()->VolumeBytes() + 0x7FF) / 0x800);
                W64(out + 8, 0);  // optical media is read-only
                W32(out + 0x10, 1);
                W32(out + 0x14, 2048);
                completed(0x18);
                return kSuccess;
            }
            W64(out, 0x100000);
            W64(out + 8, 0x80000);
            W32(out + 0x10, 32);
            W32(out + 0x14, 512);
            completed(0x18);
            return kSuccess;
        case 7:  // FileFsFullSizeInformation
            if (dvd) {
                W64(out, (MountedDisc()->VolumeBytes() + 0x7FF) / 0x800);
                W64(out + 8, 0);
                W64(out + 0x10, 0);
                W32(out + 0x18, 1);
                W32(out + 0x1C, 2048);
                completed(0x20);
                return kSuccess;
            }
            W64(out, 0x100000);
            W64(out + 8, 0x80000);
            W64(out + 0x10, 0x80000);
            W32(out + 0x18, 32);
            W32(out + 0x1C, 512);
            completed(0x20);
            return kSuccess;
        case 1:  // FileFsVolumeInformation
            W32(out + 8, 0x12345678);  // VolumeSerialNumber
            completed(0x18);
            return kSuccess;
        case 5: {  // Xbox returns the filesystem identifier as an ANSI byte string
            W32(out, 0);
            const char* name = dvd ? "XDVDFS" : "FATX";
            const uint32_t chars = dvd ? 6 : 4;
            W32(out + 4, dvd ? 255 : 42);
            W32(out + 8, chars);
            std::memcpy(HostPtr(out + 12), name, chars);
            completed(12 + chars);
            return kSuccess;
        }
        default:
            LOG_FIRST(8, "%s: class %u", who, cls);
            return kInvalidParameter;
    }
}

KERNEL_EXPORT(218, NtQueryVolumeInformationFile, 5) {
    return QueryVolumeClass(a[0], a[2], a[3], a[4], a[1], "NtQueryVolumeInformationFile");
}

// IoQueryVolumeInformation's last argument is a ULONG return length, not an IO_STATUS_BLOCK.
KERNEL_EXPORT(76, IoQueryVolumeInformation, 5) {
    return QueryVolumeClass(a[0], a[3], a[2], a[1], a[4], "IoQueryVolumeInformation", true);
}

KERNEL_EXPORT(207, NtQueryDirectoryFile, 10) {
    uint32_t h = a[0], event = a[1], apc = a[2], apc_ctx = a[3], iosb = a[4], out = a[5], len = a[6], cls = a[7],
             mask = a[8], restart = a[9];
    auto f = FileFromHandle(h);
    if (!f || !f->dir) return kInvalidHandle;
    if (cls != 1) {
        LOG_FIRST(8, "NtQueryDirectoryFile: class %u", cls);
        return kInvalidParameter;
    }
    std::string pat = mask ? ReadAnsiString(mask) : "*";
    if (!f->listed || (restart & 0xFF)) {
        f->entries.clear();
        f->disc_entries.clear();
        if (f->entry) {
            for (const Disc::Entry& e : f->entry->children)
                if (WildMatch(pat, e.name)) f->disc_entries.push_back(&e);
        } else {
            std::error_code ec;
            for (auto& e : fs::directory_iterator(f->host, ec))
                if (WildMatch(pat, e.path().filename().string())) f->entries.push_back(e);
            std::sort(f->entries.begin(), f->entries.end());
        }
        f->next = 0;
        f->listed = true;
        if (Upper(f->xbox).find("\\UDATA") != std::string::npos ||
            Upper(f->xbox).find("\\SAVEDATA") != std::string::npos) {
            LOGR("diag: save-dir-list path=%s pattern=%s host=%s entries=%zu disc=%zu restart=%08X",
                 f->xbox.c_str(), pat.c_str(), f->host.string().c_str(), f->entries.size(),
                 f->disc_entries.size(), restart);
            for (size_t j = 0; j < std::min<size_t>(f->entries.size(), 32); ++j)
                LOGR("diag: save-dir-entry path=%s name=%s dir=%u", f->xbox.c_str(),
                     f->entries[j].path().filename().string().c_str(),
                     fs::is_directory(f->entries[j]));
        }
    }
    if (f->next >= (f->entry ? f->disc_entries.size() : f->entries.size())) {
        if (Upper(f->xbox).find("\\UDATA") != std::string::npos ||
            Upper(f->xbox).find("\\SAVEDATA") != std::string::npos)
            LOGR("diag: save-dir-end path=%s next=%zu entries=%zu disc=%zu", f->xbox.c_str(), f->next,
                 f->entries.size(), f->disc_entries.size());
        IoStatus(iosb, kNoMoreFiles, 0);
        return f->next == 0 ? kNoSuchFile : kNoMoreFiles;
    }
    size_t i = f->next++;
    std::string fname = f->entry ? f->disc_entries[i]->name : f->entries[i].path().filename().string();
    if (Upper(f->xbox).find("\\UDATA") != std::string::npos ||
        Upper(f->xbox).find("\\SAVEDATA") != std::string::npos)
        LOGR("diag: save-dir-next path=%s index=%zu name=%s", f->xbox.c_str(), i, fname.c_str());
    uint32_t need = 0x40 + uint32_t(fname.size());
    if (len < need) return kBufferTooSmall;
    std::memset(HostPtr(out), 0, need);
    // times, sizes and attributes sit at the same place
    if (f->entry) FillNetworkInfo(out + 8, *f->disc_entries[i]);
    else FillNetworkInfo(out + 8, f->entries[i].path());
    // FILE_DIRECTORY_INFORMATION has EndOfFile before AllocationSize, the reverse of the network info
    W32(out + 0x38, R32(out + 8 + 0x30));
    uint64_t alloc = R64(out + 8 + 0x20), eof = R64(out + 8 + 0x28);
    W64(out + 0x28, eof);
    W64(out + 0x30, alloc);
    W32(out + 0x3C, uint32_t(fname.size()));
    std::memcpy(HostPtr(out + 0x40), fname.data(), fname.size());
    IoStatus(iosb, kSuccess, need);
    Complete(event, apc, apc_ctx, iosb);
    return kSuccess;
}

KERNEL_EXPORT(210, NtQueryFullAttributesFile, 2) {
    std::string name = ReadAnsiString(R32(a[0] + 4));
    Target t = Resolve(name, R32(a[0]));
    std::error_code ec;
    if (t.disc ? !t.entry : t.host.empty() || !fs::exists(t.host, ec)) {
        LOG_FIRST(200, "missing %s", name.c_str());
        static std::atomic<uint32_t> query_missing{0};
        uint32_t n = query_missing.fetch_add(1, std::memory_order_relaxed);
        if (t.disc && n < 64) LOGR("diag: disc-query-missing[%u] path=%s", n, name.c_str());
        return kObjectNameNotFound;
    }
    if (t.disc) {
        if (t.ca2) {
            auto decoded = DecompressCa2(*t.entry);
            if (!decoded) return kUnsuccessful;
            FillNetworkInfo(a[1], MountedDisc()->Time(), decoded->size(), false);
        } else {
            FillNetworkInfo(a[1], *t.entry);
        }
    }
    else FillNetworkInfo(a[1], t.host);
    return kSuccess;
}

KERNEL_EXPORT(195, NtDeleteFile, 1) {
    std::string host = ResolvePath(ReadAnsiString(R32(a[0] + 4)), R32(a[0]));
    std::error_code ec;
    if (host.empty() || !fs::remove(host, ec)) return kObjectNameNotFound;
    return kSuccess;
}

KERNEL_EXPORT(198, NtFlushBuffersFile, 2) {
    if (auto f = FileFromHandle(a[0]))
        if (f->fp) std::fflush(f->fp);
    IoStatus(a[1], kSuccess, 0);
    return kSuccess;
}

KERNEL_EXPORT(200, NtFsControlFile, 10) {
    IoStatus(a[4], kSuccess, 0);
    return kSuccess;
}

KERNEL_EXPORT(196, NtDeviceIoControlFile, 10) {
    uint32_t iosb = a[4], code = a[5], out = a[8], out_len = a[9];
    LOG_FIRST(16, "NtDeviceIoControlFile: code %08X", code);
    static std::atomic<uint32_t> diag_ioctls{0};
    uint32_t diag_n = diag_ioctls.fetch_add(1, std::memory_order_relaxed);
    if (diag_n < 64)
        LOGR("diag: device-ioctl[%u] caller=%08X handle=%08X code=%08X in=%u out=%u", diag_n,
             R32(ctx.esp), a[0], code, a[7], out_len);
    if (out && out_len) std::memset(HostPtr(out), 0, out_len);
    if (code == 0x70000 && out_len >= 0x18) {  // IOCTL_DISK_GET_DRIVE_GEOMETRY
        W64(out, 1000);
        W32(out + 8, 12);
        W32(out + 12, 255);
        W32(out + 16, 63);
        W32(out + 20, 512);
    }
    IoStatus(iosb, kSuccess, out_len);
    return kSuccess;
}

KERNEL_EXPORT(67, IoCreateSymbolicLink, 2) {
    CreateSymbolicLink(ReadAnsiString(a[0]), ReadAnsiString(a[1]));
    return kSuccess;
}

KERNEL_EXPORT(69, IoDeleteSymbolicLink, 1) {
    return DeleteSymbolicLink(ReadAnsiString(a[0])) ? kSuccess : kObjectNameNotFound;
}

KERNEL_EXPORT(203, NtOpenSymbolicLinkObject, 2) {
    std::string name = ObjectName(a[1]);
    std::string target = QuerySymbolicLink(name);
    if (target.empty()) return kObjectNameNotFound;
    auto l = std::make_shared<Link>();
    l->target = target;
    W32(a[0], NewHandle(l));
    return kSuccess;
}

KERNEL_EXPORT(215, NtQuerySymbolicLinkObject, 3) {
    ObjectRef o = FromHandle(a[0]);
    if (!o || o->type != ObjType::SymLink) return kInvalidHandle;
    auto& t = static_cast<Link&>(*o).target;
    uint32_t str = a[1];
    uint16_t maxlen = R16(str + 2);
    if (a[2]) W32(a[2], uint32_t(t.size()));
    if (t.size() > maxlen) return kBufferTooSmall;
    std::memcpy(HostPtr(R32(str + 4)), t.data(), t.size());
    W16(str, uint16_t(t.size()));
    return kSuccess;
}

}  // namespace kernel
