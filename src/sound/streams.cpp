// CDirectSoundStream, the object carries the XDK's own vtable
#include "sound/sound.h"
#include "xdk/layout.h"

using namespace kernel;
using namespace sound;

constexpr uint32_t kStreamVtable = SYM(DSound_StreamVtable);  // CDirectSoundStream XMediaObject vtable
constexpr uint32_t kStreamVtable2 = SYM(DSound_StreamVtable2);
constexpr uint32_t kPacketPending = 0x8000000A;  // E_PENDING
constexpr uint32_t kFlushed = 0x80004004;        // E_ABORT
static thread_local std::vector<Done> t_flushed;  // run by the STREAM call that flushed them, once g_lock is let go

// DSSTREAMDESC
constexpr uint32_t kStreamDescFlags = 0;
constexpr uint32_t kStreamDescMaxPackets = 4;
constexpr uint32_t kStreamDescFormat = 8;
constexpr uint32_t kStreamDescCallback = 12;
constexpr uint32_t kStreamDescContext = 16;
constexpr uint32_t kStreamDescMixBins = 20;

HLE_STDCALL(DirectSoundCreateStream, 2) {
    XDK_REQUIRE("dsound", XDK_KEY(CDirectSoundStream_Vtable), XDK_KEY(CDirectSoundStream_Vtable2),
                XDK_DATA(DSound_StreamVtable), XDK_DATA(DSound_StreamVtable2));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Start();
    uint32_t desc = a[0];
    uint32_t obj = mem::AllocSystem(0x80);
    std::memset(HostPtr(obj), 0, 0x80);
    W32(obj + XDK_CDirectSoundStream_Vtable, kStreamVtable);
    W32(obj + XDK_CDirectSoundStream_Vtable2, kStreamVtable2);
    Voice& v = g_voices[obj];
    v.obj = obj;
    v.stream = true;
    // Keep the guest-visible owner/state links expected by the linked DSOUND
    // property helpers. Buffers expose these same links in their synthetic
    // guest object even though the mixer voice itself is host-side.
    // The linked DSOUND property helper is called with the stream interface
    // offset by four bytes and follows its +0x10 member (stream +0x14) to a
    // state object, then reads that object's +0xB8 member. Keep that guest
    // layout valid even though the actual mixer voice lives on the host.
    v.guest_state = mem::AllocSystem(0xC0);
    std::memset(HostPtr(v.guest_state), 0, 0xC0);
    W32(v.guest_state + 0x70, g_ds);
    W32(obj + 0x0C, v.guest_state);
    W32(obj + 0x10, g_ds);
    W32(obj + 0x14, v.guest_state);
    uint32_t flags = R32(desc + kStreamDescFlags);
    v.max_packets = std::max<uint32_t>(R32(desc + kStreamDescMaxPackets), 1);
    ReadFormat(v, R32(desc + kStreamDescFormat));
    v.callback = R32(desc + kStreamDescCallback);
    v.context = R32(desc + kStreamDescContext);
    SetBins(v, R32(desc + kStreamDescMixBins));
    v.is3d = flags & kCaps3d;
    v.headroom = DefaultHeadroom(flags);
    W32(a[1], obj);
    LOGR("dsound: guest-stream interface=%08X owner=%08X state=%08X format-tag=%u rate=%u channels=%u max-packets=%u flags=%08X caller=%08X",
         obj, R32(obj + 0x10), v.guest_state, v.tag, v.rate, v.channels, v.max_packets, flags, Caller(ctx));
    return 0;
}

// The linked IDirectSound::CreateSoundStream entry receives the device as
// its first argument; the standalone host implementation takes desc and out.
// Forward calls on our synthetic device to the host stream implementation.
extern "C" void __imp__sub_003F4F91(X86Context&, uint8_t*);
X86_FUNC(sub_003F4F91) {
    const uint32_t device = R32(ctx.esp + 4);
    if (device == g_ds && device != 0 && R32(ctx.esp + 16) == 0) {
        const uint32_t desc = R32(ctx.esp + 8);
        const uint32_t out = R32(ctx.esp + 12);
        LOGR("dsound: create-stream host device=%08X desc=%08X out=%08X caller=%08X",
             device, desc, out, R32(ctx.esp));
        const uint32_t result = CallGuest(ctx, SYM(DirectSoundCreateStream), {desc, out});
        Return(ctx, result, 4);
        return;
    }
    __imp__sub_003F4F91(ctx, base);
}

// The stream variant of SetOutputBuffer is linked separately from the
// buffer variant. Host streams already mix through their own mix bins.
extern "C" void __imp__sub_003F417C(X86Context&, uint8_t*);
X86_FUNC(sub_003F417C) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint32_t obj = R32(ctx.esp + 4);
    Voice* voice = Find(obj);
    if (voice && voice->stream) {
        LOGR("dsound: stream-output host stream=%08X output=%08X caller=%08X",
             obj, R32(ctx.esp + 8), R32(ctx.esp));
        Return(ctx, 0, 2);
        return;
    }
    __imp__sub_003F417C(ctx, base);
}

// The title's linked stream flag setter accesses a hardware voice that does
// not exist on host streams. Preserve the requested flag on the host voice.
extern "C" void __imp__sub_003F3057(X86Context&, uint8_t*);
X86_FUNC(sub_003F3057) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint32_t obj = R32(ctx.esp + 4);
    Voice* voice = Find(obj);
    if (voice && voice->stream) {
        const uint32_t enabled = R32(ctx.esp + 8);
        voice->guest_stream_flag = enabled != 0;
        LOGR("dsound: stream-flag host stream=%08X enabled=%u caller=%08X", obj, enabled, R32(ctx.esp));
        Return(ctx, 0, 2);
        return;
    }
    __imp__sub_003F3057(ctx, base);
}

// The linked setter writes into the Xbox hardware voice and updates its APU.
// Keep the requested value for host streams until its mixer meaning is known.
extern "C" void __imp__sub_003F3012(X86Context&, uint8_t*);
X86_FUNC(sub_003F3012) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint32_t obj = R32(ctx.esp + 4);
    Voice* voice = Find(obj);
    if (voice && voice->stream) {
        voice->guest_stream_setting = R32(ctx.esp + 8);
        LOGR("dsound: stream-setting host stream=%08X value=%08X caller=%08X",
             obj, voice->guest_stream_setting, R32(ctx.esp));
        Return(ctx, 0, 2);
        return;
    }
    __imp__sub_003F3012(ctx, base);
}

// The linked GetMixBinVolumes walks an NV2A voice record. Host streams have
// no such record, so return equivalent values from the mixer voice. The guest
// caller expects a count followed by eight inline (bin, millibels) pairs.
extern "C" void __imp__sub_003F304D(X86Context&, uint8_t*);
X86_FUNC(sub_003F304D) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint32_t obj = R32(ctx.esp + 4);
    Voice* voice = Find(obj);
    if (voice && voice->stream) {
        const uint32_t out = R32(ctx.esp + 8);
        if (!out) {
            Return(ctx, kErrorFail, 2);
            return;
        }
        const uint32_t count = std::min<uint32_t>(voice->nbins, 8);
        W32(out, count);
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t pair = out + 4 + i * 8;
            if (i < count) {
                const uint32_t bin = voice->bin[i] & 31;
                W32(pair, bin);
                W32(pair + 4, static_cast<uint32_t>(voice->bin_mb[bin]));
            } else {
                W32(pair, 0xFFFFFFFFu);
                W32(pair + 4, 0xFFFFD8F0u);
            }
        }
        LOGR("dsound: stream-mixbin-get host stream=%08X count=%u caller=%08X",
             obj, count, R32(ctx.esp));
        Return(ctx, 0, 2);
        return;
    }
    __imp__sub_003F304D(ctx, base);
}
#define STREAM(name, nargs)                                     \
    static uint32_t str_##name(Voice& v, const Args& a);        \
    HLE_STDCALL(CDirectSoundStream_##name, nargs) {             \
        uint32_t r;                                             \
        std::vector<Done> flushed;                              \
        {                                                       \
            std::lock_guard<std::recursive_mutex> lock(g_lock); \
            Voice* v = Find(a[0]);                              \
            if (!v) Trace("miss:CDirectSoundStream_" #name, a[0], Caller(ctx), 0); \
            r = v ? str_##name(*v, a) : kErrorFail;             \
            flushed.swap(t_flushed);                            \
            Trace("CDirectSoundStream_" #name, a[0], Caller(ctx), r); \
        }                                                       \
        Run(ctx, flushed);                                      \
        return r;                                               \
    }                                                           \
    static uint32_t str_##name([[maybe_unused]] Voice& v, [[maybe_unused]] const Args& a)

// the stream's callbacks still queued and those of its flushed packets, in order, go to the calling thread
static void FlushStream(Voice& v) {
    for (auto& pk : v.packets) Complete(v, pk, kFlushed);
    v.packets.clear();
    v.pos = 0;
    for (auto it = g_done.begin(); it != g_done.end();) {
        if (it->voice != v.obj) {
            ++it;
            continue;
        }
        t_flushed.push_back(*it);
        it = g_done.erase(it);
    }
    Deactivate(v);
}

STREAM(AddRef, 1) { return uint32_t(++v.refs); }
STREAM(SetFormat, 2) {
    ReadFormat(v, a[1]);
    v.freq = 0;  // the XDK sets the pitch back to the new format's own rate
    v.block = -1;
    return 0;
}
STREAM(Release, 1) {
    if (--v.refs > 0) return uint32_t(v.refs);
    FlushStream(v);
    uint32_t obj = v.obj;
    if (v.guest_state) mem::FreeSystem(v.guest_state);
    g_voices.erase(obj);
    mem::FreeSystem(obj);
    return 0;
}
STREAM(GetInfo, 2) {
    uint32_t info = a[1];
    W32(info, 0x3);  // fixed sample size, fixed packet alignment
    W32(info + 4, v.tag == kFormatXboxAdpcm ? kAdpcmBlockBytes * v.channels : v.align);
    W32(info + 8, 0);
    W32(info + 12, 0);
    return 0;
}
STREAM(GetStatus, 2) {
    W32(a[1], v.packets.size() < v.max_packets ? 1u : 0u);  // XMEDIASTATUS_ACCEPT_INPUT
    return 0;
}
STREAM(Process, 3) {
    uint32_t in = a[1];
    if (!in) return 0;
    // XMEDIAPACKET: data, size, completed size, status, context
    Packet pk{R32(in), R32(in + 4), R32(in + 8), R32(in + 12), R32(in + 16)};
    if (pk.completed_ptr) W32(pk.completed_ptr, 0);
    if (pk.status_ptr) W32(pk.status_ptr, kPacketPending);
    v.packets.push_back(pk);
    v.active = true;
    v.ending = false;
    return 0;
}
STREAM(Discontinuity, 1) {
    if (!v.packets.empty()) v.ending = true;
    else if (v.active) FlushStream(v);
    return 0;
}
STREAM(Flush, 1) {
    FlushStream(v);
    return 0;
}
STREAM(FlushEx, 4) {
    if (a[3] && !v.active) return 0;
    FlushStream(v);
    return 0;
}
STREAM(Pause, 2) {
    v.paused = a[1] == 1;
    v.sync = a[1] == 2;
    return 0;
}
STREAM(SetVolume, 2) {
    v.volume = Db(int32_t(a[1]));
    return 0;
}
STREAM(SetFrequency, 2) {
    v.freq = a[1];
    return 0;
}
STREAM(SetFilter, 2) { return 0; }
STREAM(SetHeadroom, 2) {
    v.headroom = Db(-int32_t(a[1]));
    return 0;
}
STREAM(SetMixBinVolumes, 2) {
    SetBinVolumes(v, a[1]);
    return 0;
}
STREAM(SetMixBins, 2) {
    SetBins(v, a[1]);
    return 0;
}
STREAM(SetMaxDistance, 3) {
    v.max_dist = F32(a[1]);
    return 0;
}
STREAM(SetMinDistance, 3) {
    v.min_dist = F32(a[1]);
    return 0;
}
STREAM(SetPosition, 5) {
    for (int i = 0; i < 3; i++) v.p[i] = F32(a[1 + i]);
    return 0;
}
STREAM(SetVelocity, 5) { return 0; }
STREAM(SetI3DL2Source, 3) { return 0; }
