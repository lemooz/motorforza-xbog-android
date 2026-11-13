# reads or checks files from an Xbox disc image (XDVDFS):
#   xdvdfs.py <iso> <name> <out>   one file from the root of the image
#   xdvdfs.py --check <iso>        title id, name and xdk version of default.xbe
import struct, sys

MAGIC = b"MICROSOFT*XBOX*MEDIA"

# game partition, redump, XGD3, XGD1 video offsets; the descriptor sits at sector 32
PARTITIONS = (0, 0x18300000, 0xFD90000, 0x2080000)


def find_partition(f):
    for part in PARTITIONS:
        f.seek(part + 32 * 2048)
        if f.read(20) == MAGIC:
            return part
    return None


def read_root_table(f, base):
    root_sector, root_size = struct.unpack("<II", f.read(8))
    f.seek(base + root_sector * 2048)
    return f.read(root_size)


def walk(table, off, want):
    if off * 4 + 14 > len(table):
        return None
    left, right, sector, size, attr, nlen = struct.unpack_from("<HHIIBB", table, off * 4)
    if left == 0xFFFF:
        return None
    name = table[off * 4 + 14: off * 4 + 14 + nlen].decode("latin-1")
    if name.lower() == want:
        return sector, size
    for nxt in (left, right):
        if nxt:
            hit = walk(table, nxt, want)
            if hit:
                return hit
    return None


def check(iso):
    # title id and xdk version from default.xbe's certificate, to pick the right sdk/xdk signatures
    with open(iso, "rb") as f:
        base = find_partition(f)
        if base is None:
            sys.exit("not an Xbox disc image")
        hit = walk(read_root_table(f, base), 0, "default.xbe")
        if not hit:
            sys.exit("default.xbe not found")
        sector, size = hit
        f.seek(base + sector * 2048)
        xbe = f.read(min(size, 0x4000))  # header plus certificate
    # header: magic at 0x000, 0x100 byte signature after it, fields from 0x104
    if xbe[0x000:0x004] != b"XBEH":
        sys.exit("default.xbe has no XBEH header")
    base_addr = struct.unpack_from("<I", xbe, 0x104)[0]
    entry = struct.unpack_from("<I", xbe, 0x128)[0] ^ 0xA8FC57AB
    cert_rva = struct.unpack_from("<I", xbe, 0x118)[0]
    cert = cert_rva - base_addr
    title_id = struct.unpack_from("<I", xbe, cert + 0x08)[0]
    name = xbe[cert + 0x0C: cert + 0x0C + 80].decode("utf-16-le").split("\0")[0]
    version = struct.unpack_from("<I", xbe, cert + 0xAC)[0]
    print("partition  ", hex(base))
    print("xbe size   ", size)
    print("base addr  ", hex(base_addr))
    print("entry      ", hex(entry))
    print("title id   ", f"{title_id:08X}")
    print("title name ", name)
    print("cert version 0x%08X" % version)
    # library versions: char name[8], u32, build, minor; the build is the xdk release (5849 here)
    lib_count = struct.unpack_from("<I", xbe, 0x160)[0]
    lib_table = struct.unpack_from("<I", xbe, 0x164)[0] - base_addr
    for i in range(lib_count):
        name8, _, build, minor = struct.unpack_from("<8sIHH", xbe, lib_table + i * 16)
        lib = name8.split(b"\0")[0].decode()
        print(f"lib {lib:10} build {build} minor 0x{minor:04X}")


def extract(iso, want, out):
    with open(iso, "rb") as f:
        base = find_partition(f)
        if base is None:
            sys.exit("not an Xbox disc image")
        hit = walk(read_root_table(f, base), 0, want)
        if not hit:
            sys.exit(want + " not found")
        sector, size = hit
        f.seek(base + sector * 2048)
        data = f.read(size)
    with open(out, "wb") as o:
        o.write(data)
    print("wrote", out, size, "bytes from partition at", hex(base))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        check(sys.argv[2])
    elif len(sys.argv) == 4:
        extract(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: xdvdfs.py <iso> <name> <out>  |  xdvdfs.py --check <iso>")
