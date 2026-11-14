#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

#if defined(__ANDROID__) && defined(__aarch64__)
extern "C" void forza_fiber_swap(void* from, void* to);
#endif

#include "core/log.h"
#include "cpu/x86_ops.h"
#include "gpu/backend.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

extern "C" void __imp__sub_002823D0(X86Context&, uint8_t*);
extern "C" void __imp__sub_0025D620(X86Context&, uint8_t*);
extern "C" void __imp__sub_00285B70(X86Context&, uint8_t*);
extern "C" void __imp__sub_00168BE0(X86Context&, uint8_t*);
extern "C" void __imp__sub_002F9A32(X86Context&, uint8_t*);
extern "C" void __imp__sub_00063540(X86Context&, uint8_t*);
extern "C" void __imp__sub_00064310(X86Context&, uint8_t*);
extern "C" void __imp__sub_00099030(X86Context&, uint8_t*);
extern "C" void __imp__sub_00268A91(X86Context&, uint8_t*);
extern "C" void __imp__sub_00268B30(X86Context&, uint8_t*);
extern "C" void sub_00268B30(X86Context&, uint8_t*);
extern "C" void __imp__sub_00268A4C(X86Context&, uint8_t*);
extern "C" void sub_002FB0CC(X86Context&, uint8_t*);
extern "C" void __imp__sub_000969A0(X86Context&, uint8_t*);
extern "C" void __imp__sub_001E9890(X86Context&, uint8_t*);
extern "C" void __imp__sub_001E99B0(X86Context&, uint8_t*);
extern "C" void __imp__sub_001C8050(X86Context&, uint8_t*);
extern "C" void __imp__sub_00264E53(X86Context&, uint8_t*);
extern "C" void __imp__sub_002831F0(X86Context&, uint8_t*);
extern "C" void __imp__sub_00105000(X86Context&, uint8_t*);
extern "C" void __imp__sub_00107830(X86Context&, uint8_t*);
extern "C" void __imp__sub_00108670(X86Context&, uint8_t*);
extern "C" void __imp__sub_002F97E5(X86Context&, uint8_t*);
extern "C" void __imp__sub_00266887(X86Context&, uint8_t*);
extern "C" void __imp__sub_001103E0(X86Context&, uint8_t*);
extern "C" void __imp__sub_00110C70(X86Context&, uint8_t*);
extern "C" void __imp__sub_003F36E5(X86Context&, uint8_t*);
extern "C" void __imp__sub_003F3920(X86Context&, uint8_t*);
extern "C" void __imp__sub_003F4CBE(X86Context&, uint8_t*);
extern "C" void __imp__sub_003F8F5D(X86Context&, uint8_t*);
extern "C" void __imp__sub_003B80D0(X86Context&, uint8_t*);
extern "C" void __imp__sub_003C0D90(X86Context&, uint8_t*);
extern "C" void __imp__sub_003A90B0(X86Context&, uint8_t*);
extern "C" void __imp__sub_003A93D0(X86Context&, uint8_t*);
extern "C" void __imp__sub_003BED90(X86Context&, uint8_t*);
extern "C" void __imp__sub_00281270(X86Context&, uint8_t*);
extern "C" void __imp__sub_003A9FF0(X86Context&, uint8_t*);
extern "C" void __imp__sub_003BBF50(X86Context&, uint8_t*);
extern "C" void __imp__sub_00088060(X86Context&, uint8_t*);
extern "C" void __imp__sub_00088090(X86Context&, uint8_t*);
extern "C" void __imp__sub_0008B5E0(X86Context&, uint8_t*);
extern "C" void __imp__sub_0008B670(X86Context&, uint8_t*);
extern "C" void __imp__sub_0008B7A0(X86Context&, uint8_t*);
extern "C" void __imp__sub_000890E0(X86Context&, uint8_t*);
extern "C" void __imp__sub_000894B0(X86Context&, uint8_t*);
X86_FUNC(sub_003BBF50) {
    static std::atomic<uint32_t> null_release_calls{0};
    const uint32_t object = kernel::R32(ctx.esp + 4);
    if (!object) {
        const uint32_t n = null_release_calls.fetch_add(1, std::memory_order_relaxed);
        if (n < 16 || (n + 1) % 1000 == 0)
            LOGR("xdk: null release skipped[%u] caller=%08X", n + 1, kernel::R32(ctx.esp));
        kernel::Return(ctx, 0, 1);
        return;
    }
    __imp__sub_003BBF50(ctx, base);
}

#if defined(__ANDROID__) && defined(__aarch64__)
namespace {
struct FiberHostContext {
    uint64_t callee_saved[12];  // x19-x30, kept in sync with native/fiber_context.S
    uintptr_t sp;
};
struct HostFiber {
    FiberHostContext host{};
    std::vector<uint8_t> stack;
    X86Context* guest = nullptr;
    uint8_t* base = nullptr;
    uint32_t first_target = 0;
    uint32_t first_esp = 0;
    uint32_t parent = 0;
    bool initialized = false;
    bool finished = false;
};
thread_local std::unordered_map<uint32_t, std::unique_ptr<HostFiber>> g_host_fibers;
thread_local HostFiber* g_starting_fiber = nullptr;
thread_local HostFiber* g_current_host_fiber = nullptr;

extern "C" [[noreturn]] void forza_fiber_start() {
    HostFiber* f = g_starting_fiber;
    g_starting_fiber = nullptr;
    if (!f || !f->guest || !f->base || !f->first_target) std::abort();
    g_current_host_fiber = f;
    // An Xbox fiber's startup thunk is entered with an EBP based on the
    // synthetic guest stack frame. x86_resume only changes EIP/ESP, so the
    // host thread's previous EBP (often zero) would otherwise be reused.
    f->guest->ebp = f->first_esp;
    LOGR("diag: fiber-entry guest=%08X target=%08X esp=%08X ebp=%08X words=%08X/%08X/%08X/%08X/%08X",
         f->guest->esi, f->first_target, f->first_esp, f->guest->ebp,
         kernel::R32(f->first_esp - 8), kernel::R32(f->first_esp - 4),
         kernel::R32(f->first_esp), kernel::R32(f->first_esp + 4),
         kernel::R32(f->first_esp + 8));
    x86_resume(*f->guest, f->base, {f->first_target, f->first_esp});
    LOGE("fiber: guest entry %08X returned without switching fibers", f->first_target);
    std::abort();
}
}  // namespace
#endif

namespace {
std::atomic<uint32_t> g_queue_init_calls{0};
std::atomic<uint32_t> g_guest_alloc_calls{0};
std::atomic<uint32_t> g_heap_alloc_calls{0};
std::atomic<uint32_t> g_init_calls{0};
std::atomic<uint32_t> g_main_calls{0};
std::atomic<uint32_t> g_clock_update_calls{0};
std::atomic<uint32_t> g_clock_setup_calls{0};
std::atomic<uint32_t> g_format_calls{0};
std::atomic<uint32_t> g_wait_calls{0};
std::atomic<uint32_t> g_stream_calls{0};
std::atomic<uint32_t> g_asset_lookup_calls{0};
std::atomic<uint32_t> g_asset_compare_calls{0};
std::atomic<uint32_t> g_asset_factory_calls{0};
std::atomic<uint32_t> g_asset_parse_calls{0};
uint32_t g_empty_guest_string = 0;
std::atomic<uint32_t> g_io_callback_calls{0};
std::atomic<uint32_t> g_resource_walk_calls{0};
std::atomic<uint32_t> g_com_init_calls{0};
std::atomic<uint32_t> g_com_ctor_calls{0};
std::atomic<uint32_t> g_com_factory_calls{0};
std::atomic<uint32_t> g_fatal_dvd_log_calls{0};
std::atomic<uint32_t> g_ftm_open_calls{0};
std::atomic<uint32_t> g_audio_mixer_calls{0};
std::atomic<uint32_t> g_event_dispatch_calls{0};
std::atomic<uint32_t> g_event_queue_init_calls{0};
std::atomic<uint32_t> g_bink_video_update_calls{0};
std::atomic<uint32_t> g_bink_video_stop_calls{0};
std::atomic<uint32_t> g_video_overlay_calls{0};
std::atomic<uint32_t> g_work_queue_polls{0};
std::atomic<uint32_t> g_title_callback_calls{0};
}

// This zero-return guest callback is the last indirect call repeatedly observed
// on the title thread after the Bink preview. Capture its call site and arguments
// to determine which state machine is polling it, without changing its result.
X86_FUNC(sub_003A9FF0) {
    const uint32_t n = g_title_callback_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 32) {
        LOGR("diag: title-callback[%u] enter thread=%X caller=%08X arg0=%08X arg1=%08X ecx=%08X edx=%08X object=%08X active=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp + 8), ctx.ecx, ctx.edx,
             kernel::R32(0x005E5D90), kernel::R32(0x005E5D80));
    }
    __imp__sub_003A9FF0(ctx, base);
    if (n < 32)
        LOGR("diag: title-callback[%u] exit eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_0025D620) {
    const uint32_t n = g_fatal_dvd_log_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t message = kernel::R32(ctx.esp + 4);
    if (n < 64) {
        const std::string text = message >= 0x10000 ? kernel::ReadCString(message, 192) : std::string();
        LOGR("diag: fatal-dvd-log[%u] thread=%X caller=%08X message=%08X text='%s'",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), message, text.c_str());
    }
    __imp__sub_0025D620(ctx, base);
    if (n < 64) LOGR("diag: fatal-dvd-log[%u] returned eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_00285B70) {
    const uint32_t n = g_ftm_open_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t path = kernel::R32(ctx.esp + 4);
    if (n < 32) {
        const std::string text = path >= 0x10000 ? kernel::ReadCString(path, 160) : std::string();
        LOGR("diag: ftm-open[%u] thread=%X caller=%08X object=%08X path=%08X text='%s'",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), ctx.ecx, path, text.c_str());
    }
    __imp__sub_00285B70(ctx, base);
    if (n < 32) {
        const uint32_t object = ctx.eax;
        if (object >= 0x10000)
            LOGR("diag: ftm-open[%u] result=%08X handle=%08X size=%08X next=%08X block=%08X buf0=%08X buf1=%08X",
                 n, object, kernel::R32(object + 0x120), kernel::R32(object + 0x124),
                 kernel::R32(object + 0x128), kernel::R32(object + 0x288),
                 kernel::R32(object + 0x280), kernel::R32(object + 0x284));
        else LOGR("diag: ftm-open[%u] result=%08X", n, object);
    }
}

X86_FUNC(sub_003F8F5D) {
    const uint32_t n = g_audio_mixer_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = ctx.ecx;
    const uint32_t a0 = kernel::R32(ctx.esp + 4), a1 = kernel::R32(ctx.esp + 8);
    if (n < 64) {
        const uint32_t state = object >= 0x10000 ? kernel::R32(object + 0x70) : 0;
        const uint32_t pB4 = state >= 0x10000 ? kernel::R32(state + 0xB4) : 0;
        const uint32_t pB8 = state >= 0x10000 ? kernel::R32(state + 0xB8) : 0;
        LOGR("diag: audio-mixer[%u] enter thread=%X ret=%08X this=%08X args=%08X,%08X state=%08X state+b4=%08X state+b8=%08X mode=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             object, a0, a1, state, pB4, pB8, kernel::R32(0x40FF7C));
    }
    __imp__sub_003F8F5D(ctx, base);
    if (n < 64) LOGR("diag: audio-mixer[%u] exit eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_003B80D0) {
    const uint32_t n = g_event_dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = ctx.ecx;
    const uint32_t arg1 = kernel::R32(ctx.esp + 4), arg2 = kernel::R32(ctx.esp + 8);
    const uint32_t p794 = object >= 0x10000 ? kernel::R32(object + 0x794) : 0;
    const uint32_t p794flags = p794 >= 0x10000 ? kernel::R32(p794 + 4) : 0;
    if (n < 64)
        LOGR("diag: event-dispatch[%u] enter thread=%X ret=%08X this=%08X args=%08X,%08X p794=%08X p794flags=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             object, arg1, arg2, p794, p794flags);
    __imp__sub_003B80D0(ctx, base);
    if (n < 64) LOGR("diag: event-dispatch[%u] exit eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_003C0D90) {
    const uint32_t n = g_event_queue_init_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = ctx.ecx;
    const uint32_t arg0 = kernel::R32(ctx.esp + 4);
    if (n < 16)
        LOGR("diag: event-queue-init[%u] enter thread=%X ret=%08X this=%08X arg=%08X field794=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), object,
             arg0, object >= 0x10000 ? kernel::R32(object + 0x794) : 0);
    __imp__sub_003C0D90(ctx, base);
    if (n < 16)
        LOGR("diag: event-queue-init[%u] exit result=%08X field794=%08X field798=%08X",
             n, ctx.eax, object >= 0x10000 ? kernel::R32(object + 0x794) : 0,
             object >= 0x10000 ? kernel::R32(object + 0x798) : 0);
}

X86_FUNC(sub_003F4CBE) {
    const uint32_t n = g_com_factory_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t a0 = kernel::R32(ctx.esp + 4), a1 = kernel::R32(ctx.esp + 8);
    const uint32_t a2 = kernel::R32(ctx.esp + 12), a3 = kernel::R32(ctx.esp + 16);
    if (n < 32)
        LOGR("diag: com-factory[%u] enter ret=%08X args=%08X,%08X,%08X,%08X mode=%08X",
             n, kernel::R32(ctx.esp), a0, a1, a2, a3, kernel::R32(0x40FF84));
    __imp__sub_003F4CBE(ctx, base);
    if (n < 32)
        LOGR("diag: com-factory[%u] exit eax=%08X out=%08X", n, ctx.eax,
             a1 >= 0x10000 ? kernel::R32(a1) : 0);
}

X86_FUNC(sub_003F3920) {
    const uint32_t n = g_com_ctor_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = ctx.ecx;
    if (n < 32)
        LOGR("diag: com-ctor[%u] enter this=%08X arg=%08X ret=%08X",
             n, object, kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp));
    __imp__sub_003F3920(ctx, base);
    if (n < 32)
        LOGR("diag: com-ctor[%u] exit result=%08X vtable=%08X refs=%08X p08=%08X p0c=%08X p10=%08X p18=%08X",
             n, ctx.eax, kernel::R32(object), kernel::R32(object + 4), kernel::R32(object + 8),
             kernel::R32(object + 0x0C), kernel::R32(object + 0x10), kernel::R32(object + 0x18));
}

X86_FUNC(sub_003F36E5) {
    const uint32_t n = g_com_init_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t a0 = kernel::R32(ctx.esp + 4);
    const uint32_t a1 = kernel::R32(ctx.esp + 8);
    const uint32_t a2 = kernel::R32(ctx.esp + 12);
    const uint32_t p10 = a0 >= 0x10000 ? kernel::R32(a0 + 0x10) : 0;
    const uint32_t pb4 = p10 >= 0x10000 ? kernel::R32(p10 + 0xB4) : 0;
    if (n < 32)
        LOGR("diag: com-init[%u] ret=%08X args=%08X,%08X,%08X global=%08X p10=%08X pB4=%08X",
             n, kernel::R32(ctx.esp), a0, a1, a2, kernel::R32(0x40FF84), p10, pb4);
    __imp__sub_003F36E5(ctx, base);
    if (n < 32) LOGR("diag: com-init[%u] result=%08X", n, ctx.eax);
}

X86_FUNC(sub_00108670) {
    const uint32_t n = g_asset_parse_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t input = kernel::R32(ctx.esp + 4);
    const uint32_t arg1 = kernel::R32(ctx.esp + 8);
    const uint32_t arg2 = kernel::R32(ctx.esp + 12);
    if (n < 32) {
        const std::string text = input >= 0x10000 ? kernel::ReadCString(input, 96) : std::string();
        LOGR("diag: asset-parse[%u] enter thread=%X ret=%08X input=%08X text='%s' arg1=%08X arg2=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), input, text.c_str(), arg1, arg2);
    }
    __imp__sub_00108670(ctx, base);
    if (n < 32) LOGR("diag: asset-parse[%u] exit eax=%08X cf=%u zf=%u sf=%u", n, ctx.eax, ctx.cf, ctx.zf, ctx.sf);
}

X86_FUNC(sub_00266887) {
    const uint32_t thread = kernel::R32(ctx.fs_base + 0x28);
    const uint32_t data = kernel::R32(thread + 0x44);
    const uint32_t value = kernel::R32(ctx.esp + 4);
    LOGR("diag: thread-local-store thread=%08X data=%08X field14=%08X value=%08X ret=%08X",
         thread, data, data >= 0x10000 ? kernel::R32(data + 0x14) : 0, value, kernel::R32(ctx.esp));
    __imp__sub_00266887(ctx, base);
}

X86_FUNC(sub_001103E0) {
    const uint32_t n = g_io_callback_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 64)
        LOGR("diag: io-callback[%u] enter thread=%X ret=%08X a1=%08X a2=%08X a3=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp + 8), kernel::R32(ctx.esp + 12));
    __imp__sub_001103E0(ctx, base);
    if (n < 64) LOGR("diag: io-callback[%u] exit eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_00110C70) {
    const uint32_t n = g_resource_walk_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 64)
        LOGR("diag: resource-walk[%u] enter thread=%X this=%08X arg1=%08X arg2=%08X ret=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, ctx.ecx,
             kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp + 8), kernel::R32(ctx.esp));
    __imp__sub_00110C70(ctx, base);
    if (n < 64) LOGR("diag: resource-walk[%u] exit eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_00107830) {
    const uint32_t n = g_asset_factory_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t name = kernel::R32(ctx.esp + 4);
    const uint32_t key = kernel::R32(ctx.esp + 8);
    if (n < 64)
        LOGR("diag: asset-factory[%u] ret=%08X name=%08X key=%08X context=%08X",
             n, kernel::R32(ctx.esp), kernel::R32(ctx.esp + 4), key, kernel::R32(ctx.esp + 12));
    if (name < 0x10000) {
        if (!g_empty_guest_string) {
            g_empty_guest_string = mem::AllocSystem(1);
            kernel::G<uint8_t>(g_empty_guest_string) = 0;
        }
        LOGR("diag: asset-factory-invalid-name name=%08X replaced-by-empty=%08X", name, g_empty_guest_string);
        kernel::W32(ctx.esp + 4, g_empty_guest_string);
    }
    __imp__sub_00107830(ctx, base);
}

X86_FUNC(sub_00105000) {
    const uint32_t n = g_asset_lookup_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t key = kernel::R32(ctx.esp + 4);
    if (key < 0x10000) {
        LOGR("diag: asset-lookup-invalid-key this=%08X key=%08X ret=%08X; report not-found",
             ctx.ecx, key, kernel::R32(ctx.esp));
        ctx.eax = 0x0000FFFF;
        ctx.esp += 8;  // thiscall callee pops its single stack argument
        return;
    }
    if (n < 32)
        LOGR("diag: asset-lookup[%u] this=%08X begin=%08X end=%08X key=%08X ret=%08X",
             n, ctx.ecx, kernel::R32(ctx.ecx + 0x90), kernel::R32(ctx.ecx + 0x94),
             key, kernel::R32(ctx.esp));
    __imp__sub_00105000(ctx, base);
    if (n < 32) LOGR("diag: asset-lookup[%u] result=%08X", n, ctx.eax);
}

X86_FUNC(sub_002F97E5) {
    const uint32_t n = g_asset_compare_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 64)
        LOGR("diag: asset-compare[%u] left=%08X right=%08X ret=%08X unicode=%08X",
             n, kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp + 8), kernel::R32(ctx.esp),
             kernel::R32(0x005E71E4));
    __imp__sub_002F97E5(ctx, base);
}

X86_FUNC(sub_002831F0) {
    const uint32_t n = g_stream_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = ctx.ecx;
    __imp__sub_002831F0(ctx, base);
    const uint8_t failed = kernel::G<uint8_t>(object + 0x28);
    if (n < 32 || failed)
        LOGR("diag: asset-stream[%u] failed=%u handle=%08X base=%08X total=%u cursor=%u chunk=%u buffer=%08X result=%08X",
             n, failed, kernel::R32(object), kernel::R32(object + 4), kernel::R32(object + 8),
             kernel::R32(object + 0x0C), kernel::R32(object + 0x14), kernel::R32(object + 0x1C), ctx.eax);
}

X86_FUNC(sub_001C8050) {
    const uint32_t n = g_format_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t caller = kernel::R32(ctx.esp);
    const auto start = std::chrono::steady_clock::now();
    __imp__sub_001C8050(ctx, base);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (n < 12 || us > 10000)
        LOGR("diag: init-format[%u] caller=%08X duration-us=%lld", n, caller, (long long)us);
}

X86_FUNC(sub_00264E53) {
    const uint32_t n = g_wait_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t caller = kernel::R32(ctx.esp);
    const auto start = std::chrono::steady_clock::now();
    __imp__sub_00264E53(ctx, base);
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    if (n < 12 || us > 10000)
        LOGR("diag: init-wait[%u] caller=%08X duration-us=%lld result=%08X", n, caller, (long long)us, ctx.eax);
}

X86_FUNC(sub_001E9890) {
    uint32_t n = g_clock_update_calls.fetch_add(1, std::memory_order_relaxed);
    const bool sample = n < 24 || (n % 120) == 0;
    if (sample)
        LOGR("diag: frame-clock[%u] enter thread=%X ret=%08X arg=%u count=%u enabled=%u ready=%u period=%u",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(ctx.esp + 4), kernel::R32(0x005E0370), kernel::G<uint8_t>(0x005E0359),
             kernel::G<uint8_t>(0x005E0358), kernel::R32(0x005696E8));
    __imp__sub_001E9890(ctx, base);
    if (sample)
        LOGR("diag: frame-clock[%u] exit count=%u enabled=%u ready=%u elapsed=%u tick=%u phase=%u prev=%u time=%u",
             n, kernel::R32(0x005E0370), kernel::G<uint8_t>(0x005E0359), kernel::G<uint8_t>(0x005E0358),
             kernel::R32(0x005E0330), kernel::R32(0x005E0364), kernel::R32(0x005E0368),
             kernel::R32(0x005E036C), kernel::R32(0x58DB68));
}

X86_FUNC(sub_001E99B0) {
    uint32_t n = g_clock_setup_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 12)
        LOGR("diag: frame-clock-setup[%u] thread=%X ret=%08X count=%u period=%u",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(0x005E0370), kernel::R32(0x005696E8));
    __imp__sub_001E99B0(ctx, base);
    if (n < 12)
        LOGR("diag: frame-clock-setup[%u] exit count=%u period=%u ready=%u",
             n, kernel::R32(0x005E0370), kernel::R32(0x005696E8), kernel::G<uint8_t>(0x005E0358));
}

// The generated wrappers are weak; these strong wrappers preserve the original guest
// implementation while exposing the queue/allocator path in logcat.
X86_FUNC(sub_002823D0) {
    uint32_t n = g_queue_init_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 16)
        LOGR("diag: queue-init[%u] enter thread=%X ret=%08X old=%08X heap=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(0x005E5D90), kernel::R32(0x005E7360));
    __imp__sub_002823D0(ctx, base);
    if (n < 16)
        LOGR("diag: queue-init[%u] exit eax=%08X object=%08X", n, ctx.eax, kernel::R32(0x005E5D90));
}

X86_FUNC(sub_00168BE0) {
    uint32_t size = kernel::R32(ctx.esp + 4);
    uint32_t n = size == 12 ? g_guest_alloc_calls.fetch_add(1, std::memory_order_relaxed) : 16;
    if (n < 16)
        LOGR("diag: guest-alloc[%u] size=%u caller=%08X source=%08X tag=%08X",
             n, size, kernel::R32(ctx.esp), kernel::R32(ctx.esp + 8), kernel::R32(ctx.esp + 12));
    __imp__sub_00168BE0(ctx, base);
    if (n < 16) LOGR("diag: guest-alloc[%u] -> eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_002F9A32) {
    uint32_t size = kernel::R32(ctx.esp + 4);
    uint32_t n = size == 12 ? g_heap_alloc_calls.fetch_add(1, std::memory_order_relaxed) : 16;
    if (n < 16) LOGR("diag: heap-alloc[%u] size=%u heap=%08X mode=%u", n, size,
                      kernel::R32(0x005E7360), kernel::R32(0x005EBD80));
    __imp__sub_002F9A32(ctx, base);
    if (n < 16) LOGR("diag: heap-alloc[%u] -> eax=%08X", n, ctx.eax);
}

X86_FUNC(sub_00063540) {
    uint32_t n = g_init_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 8)
        LOGR("diag: title-init[%u] enter thread=%X ret=%08X heap=%08X queue=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(0x005E7360), kernel::R32(0x005E5D90));
    __imp__sub_00063540(ctx, base);
    if (n < 8)
        LOGR("diag: title-init[%u] exit eax=%08X heap=%08X queue=%08X",
             n, ctx.eax, kernel::R32(0x005E7360), kernel::R32(0x005E5D90));
}

X86_FUNC(sub_00064310) {
    uint32_t n = g_main_calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 8)
        LOGR("diag: title-main[%u] enter thread=%X ret=%08X heap=%08X queue=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp),
             kernel::R32(0x005E7360), kernel::R32(0x005E5D90));
    __imp__sub_00064310(ctx, base);
}

X86_FUNC(sub_00099030) {
    static std::atomic<uint32_t> calls{0};
    const uint32_t n = calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t index = kernel::R32(0x005E4C28);
    const uint32_t stack_base = kernel::R32(ctx.fs_base + 4);
    const uint32_t slot = stack_base + index * 4;
    const bool in_stack = slot >= mem::kStackBase && slot < mem::kStackBase + mem::kStackSize;
    const uint32_t value = in_stack ? kernel::R32(slot) : 0xFFFFFFFF;
    if (n < 16)
        LOGR("diag: xbe-tls-use[%u] thread=%X caller=%08X self=%08X fs=%08X index=%08X stack=%08X slot=%08X slot-value=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), ctx.ecx,
             ctx.fs_base, index, stack_base, slot, value);
    if (ctx.ecx < 0x10000) {
        LOGR("diag: null-guest-destructor thread=%X caller=%08X self=%08X slot=%08X value=%08X; skipped",
             kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), ctx.ecx, slot, value);
        ctx.eax = 0;
        return;
    }
    __imp__sub_00099030(ctx, base);
}

X86_FUNC(sub_00268A91) {
    __imp__sub_00268A91(ctx, base);
#if defined(__ANDROID__) && defined(__aarch64__)
    // The guest allocator can reuse a deleted fiber's address. Its new
    // startup stack must not inherit the previous host fiber's finished flag.
    if (ctx.eax) {
        auto old = g_host_fibers.find(ctx.eax);
        if (old != g_host_fibers.end() && old->second.get() != g_current_host_fiber) {
            LOGR("diag: fiber-create-reuse guest=%08X finished=%u", ctx.eax, old->second->finished);
            g_host_fibers.erase(old);
        }
    }
#endif
}

X86_FUNC(sub_00268B30) {
    static std::atomic<uint32_t> calls{0};
    const uint32_t n = calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t arg = kernel::R32(ctx.esp + 4);
    if (auto done = g_host_fibers.find(arg); done != g_host_fibers.end() && done->second->finished) {
        // A completed Xbox fiber is no longer a valid switch target. Treat
        // stale scheduler attempts as a return to the caller instead of
        // re-entering its finished startup thunk.
        ctx.esp += 8;  // ret 4
        if (n < 16) LOGR("diag: fiber-switch[%u] skip-finished self=%08X arg=%08X", n, ctx.esi, arg);
        return;
    }
    const uint32_t index = kernel::R32(0x005E4C28);
    const uint32_t slot = kernel::R32(ctx.fs_base + 4) + index * 4;
    const uint32_t tls_object = slot >= mem::kStackBase && slot < mem::kStackBase + mem::kStackSize
                                    ? kernel::R32(slot) : 0xFFFFFFFF;
    const uint32_t current_fiber = tls_object >= mem::kStackBase && tls_object < mem::kStackBase + mem::kStackSize
                                       ? kernel::R32(tls_object + 0x0C) : 0;
    if (n < 16)
        LOGR("diag: fiber-switch[%u] before self=%08X arg=%08X esp=%08X tls-slot=%08X object=%08X object-field0c=%08X",
             n, ctx.esi, arg, ctx.esp, slot, tls_object,
             tls_object >= mem::kStackBase && tls_object < mem::kStackBase + mem::kStackSize
                 ? kernel::R32(tls_object + 0x0C) : 0xFFFFFFFF);
    __imp__sub_00268B30(ctx, base);
#if defined(__ANDROID__) && defined(__aarch64__)
    // Keep the native C++ call stack paired with the guest fiber stack. The
    // generated switch already restored the target guest registers and ESP.
    static thread_local uint32_t switches = 0;
    if (current_fiber && arg && current_fiber != arg) {
        auto& from = g_host_fibers[current_fiber];
        if (!from) from = std::make_unique<HostFiber>();
        auto& to = g_host_fibers[arg];
        if (!to) to = std::make_unique<HostFiber>();
        if (!to->initialized) {
            to->stack.resize(8u * 1024u * 1024u);
            to->guest = &ctx;
            to->base = base;
            const uint32_t resume_sp = ctx.esp - 8;
            to->first_target = kernel::R32(resume_sp);
            to->first_esp = ctx.esp;
            to->parent = current_fiber;
            to->host.sp = (reinterpret_cast<uintptr_t>(to->stack.data() + to->stack.size())) & ~uintptr_t(15);
            to->host.callee_saved[0] = reinterpret_cast<uintptr_t>(to.get());  // x19, passed to the entry trampoline
            to->host.callee_saved[11] = reinterpret_cast<uintptr_t>(&forza_fiber_start);  // x30
            to->initialized = true;
            g_starting_fiber = to.get();
            if (switches < 16)
                LOGR("diag: host-fiber[%u] create guest=%08X target=%08X esp=%08X stack=%zu",
                     switches, arg, to->first_target, to->first_esp, to->stack.size());
        } else {
            to->guest = &ctx;
            to->base = base;
        }
        switches++;
        from->initialized = true;  // forza_fiber_swap writes the live native stack here
        g_current_host_fiber = to.get();
        forza_fiber_swap(&from->host, &to->host);
        g_current_host_fiber = from.get();
        if (switches < 32)
            LOGR("diag: host-fiber resume current=%08X esp=%08X", current_fiber, ctx.esp);
    }
#endif
    if (n < 16) {
        const uint32_t resume_sp = ctx.esp - 8;
        LOGR("diag: fiber-switch[%u] after self=%08X eax=%08X esp=%08X ret=%08X arg=%08X tls-slot-value=%08X",
             n, ctx.esi, ctx.eax, ctx.esp, kernel::R32(resume_sp), kernel::R32(resume_sp + 4),
             slot >= mem::kStackBase && slot < mem::kStackBase + mem::kStackSize ? kernel::R32(slot) : 0xFFFFFFFF);
    }
}

X86_FUNC(sub_00268A4C) {
    X86_LOAD_FLAGS();
    LOGR("diag: fiber-start-thunk enter thread=%X esp=%08X ebp=%08X esi=%08X stack=%08X/%08X/%08X/%08X",
         kernel::Current() ? kernel::Current()->id : 0, ctx.esp, ctx.ebp, ctx.esi,
         kernel::R32(ctx.esp), kernel::R32(ctx.esp + 4), kernel::R32(ctx.esp + 8),
         kernel::R32(ctx.esp + 12));
    // Rebuild the Xbox fiber startup thunk with an explicit frame-pointer
    // save around the user callback. The callback may switch fibers, and the
    // shared X86Context can carry another fiber's EBP when it resumes.
    X86_PUSH32(0x08u);
    X86_PUSH32(0x004CE6E0u);
    X86_PUSH32(0x00268A58u);
    sub_002FB0CC(ctx, base);
    X86_MEM32(ctx.ebp - 4) &= 0;
    ctx.eax = X86_MEM32(0x005E4C28u);
    ctx.ecx = X86_MEM32(ctx.fs_base + 4);
    ctx.eax = X86_MEM32(ctx.ecx + ctx.eax * 4u);
    ctx.eax = X86_MEM32(ctx.eax + 0x0Cu);
    const uint32_t startup_context = X86_MEM32(ctx.eax);
    X86_PUSH32(startup_context);
    const uint32_t callback = X86_MEM32(ctx.ebp + 8);
    const uint32_t frame_ebp = ctx.ebp;
    const uint32_t startup_vtable = startup_context >= 0x10000u ? X86_MEM32(startup_context) : 0;
    LOGR("diag: fiber-start-call target=%08X context=%08X vtable=%08X method0c=%08X method3c=%08X frame=%08X",
         callback, startup_context, startup_vtable,
         startup_vtable ? X86_MEM32(startup_vtable + 0x0Cu) : 0,
         startup_vtable ? X86_MEM32(startup_vtable + 0x3Cu) : 0, frame_ebp);
    X86_PUSH32(0x00268A76u);
    if (callback < 0x10000u)
        LOGE("fiber: invalid startup callback=%08X frame=%08X stack=%08X/%08X",
             callback, frame_ebp, X86_MEM32(frame_ebp + 8), X86_MEM32(frame_ebp + 12));
    else
        x86_icall(ctx, base, callback);
    ctx.ebp = frame_ebp;
    LOGR("diag: fiber-start-thunk callback-return target=%08X esp=%08X ebp=%08X eax=%08X",
         callback, ctx.esp, ctx.ebp, ctx.eax);
    X86_MEM32(ctx.ebp - 4) |= 0xFFFFFFFFu;
    if (g_current_host_fiber && g_current_host_fiber->parent) {
        const uint32_t parent = g_current_host_fiber->parent;
        g_current_host_fiber->finished = true;
        LOGR("diag: fiber-start-thunk complete guest=%08X return-to=%08X esp=%08X",
             g_current_host_fiber->guest ? g_current_host_fiber->guest->esi : 0, parent, ctx.esp);
        X86_PUSH32(parent);
        X86_PUSH32(0x00268A8Au);
        sub_00268B30(ctx, base);
    }
    X86_PUSH32(0u);
    x86_trap(ctx, 0x00268A90u, "int3");
}

X86_FUNC(sub_000969A0) {
    const uint32_t object = kernel::R32(ctx.esp + 4);
    const uint32_t vtable = object >= 0x10000u ? kernel::R32(object) : 0;
    const uint32_t method3c = vtable ? kernel::R32(vtable + 0x3C) : 0;
    const uint32_t method0c = vtable ? kernel::R32(vtable + 0x0C) : 0;
    LOGR("diag: fiber-start-callback enter object=%08X vtable=%08X method3c=%08X method0c=%08X arg2=%08X",
         object, vtable, method3c, method0c, kernel::R32(ctx.esp + 8));
    __imp__sub_000969A0(ctx, base);
    LOGR("diag: fiber-start-callback return object=%08X eax=%08X esp=%08X ebp=%08X",
         object, ctx.eax, ctx.esp, ctx.ebp);
}

// Bink's game-side video update decodes a frame and copies its planes into the
// title's video surfaces. Record whether that copy path runs and whether the
// decoder's frame index advances, without changing the guest implementation.
X86_FUNC(sub_003A90B0) {
    const uint32_t n = g_bink_video_update_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t self = ctx.ecx;
    const uint32_t bink = self >= 0x10000 ? kernel::R32(self + 0x48) : 0;
    if (n < 16 && bink >= 0x10000)
        LOGR("diag: bink-video[%u] enter self=%08X bink=%08X frame=%u/%u field10=%08X video-flag=%u",
             n, self, bink, kernel::R32(bink + 0x10), kernel::R32(bink + 8),
             kernel::R32(self + 0x10), kernel::G<uint8_t>(0x0056D890));
    __imp__sub_003A90B0(ctx, base);
    if (n < 16 && bink >= 0x10000)
        LOGR("diag: bink-video[%u] exit result=%08X frame=%u/%u", n, ctx.eax,
             kernel::R32(bink + 0x10), kernel::R32(bink + 8));
}

// The title invokes this callback while stopping/resetting its video object.
// Capture the object's transition to see whether the black frame is caused by
// video teardown or whether the frontend never takes over afterward.
X86_FUNC(sub_003A93D0) {
    const uint32_t n = g_bink_video_stop_calls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t self = ctx.ecx;
    if (n < 32)
        LOGR("diag: bink-stop[%u] enter thread=%X ret=%08X self=%08X p04=%08X p0c=%08X p10=%08X video-flag=%u",
             n, kernel::Current() ? kernel::Current()->id : 0, kernel::R32(ctx.esp), self,
             self >= 0x10000 ? kernel::R32(self + 4) : 0,
             self >= 0x10000 ? kernel::R32(self + 0x0C) : 0,
             self >= 0x10000 ? kernel::R32(self + 0x10) : 0,
             kernel::G<uint8_t>(0x0056D890));
    __imp__sub_003A93D0(ctx, base);
    gpu::g_capture_front_after_input.store(true, std::memory_order_release);
    if (n < 32)
        LOGR("diag: bink-stop[%u] exit eax=%08X p04=%08X p0c=%08X p10=%08X video-flag=%u",
             n, ctx.eax, self >= 0x10000 ? kernel::R32(self + 4) : 0,
             self >= 0x10000 ? kernel::R32(self + 0x0C) : 0,
             self >= 0x10000 ? kernel::R32(self + 0x10) : 0,
             kernel::G<uint8_t>(0x0056D890));
}

// The Xbox title sends Bink frames through NV2A's PVIDEO overlay registers,
// independently of the normal D3D draw calls. Log the programmed overlay state
// to establish whether the game actually submits frames for presentation.
X86_FUNC(sub_003BED90) {
    const uint32_t n = g_video_overlay_calls.fetch_add(1, std::memory_order_relaxed);
    __imp__sub_003BED90(ctx, base);
    if (n < 32) {
        const uint32_t mmio = kernel::R32(kernel::R32(0x003C8A48) + 0x934);
        const uint32_t frame = mem::VirtualFromPhysical(kernel::R32(mmio + 0x8900) + kernel::R32(mmio + 0x8920));
        uint32_t hash = 2166136261u;
        if (n < 8 && mem::IsCommitted(frame) && mem::IsCommitted(frame + 640u * 480u * 2u - 1)) {
            const auto* bytes = static_cast<const uint8_t*>(kernel::HostPtr(frame));
            for (uint32_t i = 0; i < 640u * 480u * 2u; i += 257u) hash = (hash ^ bytes[i]) * 16777619u;
        }
        LOGR("diag: bink-overlay[%u] thread=%X mmio=%08X buffer=%08X base=%08X limit=%08X offset=%08X size-in=%08X point-in=%08X ds/dt=%08X/%08X point-out=%08X size-out=%08X format=%08X",
             n, kernel::Current() ? kernel::Current()->id : 0, mmio,
             kernel::R32(mmio + 0x8700), kernel::R32(mmio + 0x8900), kernel::R32(mmio + 0x8908),
             kernel::R32(mmio + 0x8920), kernel::R32(mmio + 0x8928), kernel::R32(mmio + 0x8930),
             kernel::R32(mmio + 0x8938), kernel::R32(mmio + 0x8940), kernel::R32(mmio + 0x8948),
             kernel::R32(mmio + 0x8950), kernel::R32(mmio + 0x8958));
        if (n < 8 && mem::IsCommitted(frame) && mem::IsCommitted(frame + 640u * 480u * 2u - 1)) {
            const auto* bytes = static_cast<const uint8_t*>(kernel::HostPtr(frame));
            LOGR("diag: bink-overlay-data[%u] addr=%08X sample-hash=%08X first=%02X%02X%02X%02X%02X%02X%02X%02X",
                 n, frame, hash, bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
        }
    }
}

// The title's main loop spends most of its life polling this asynchronous work
// queue. Sample its real head/count as it transitions from FMV to frontend.
X86_FUNC(sub_00281270) {
    const uint32_t n = g_work_queue_polls.fetch_add(1, std::memory_order_relaxed);
    const uint32_t object = kernel::R32(0x005E5D90);
    if (n < 16 || n % 600 == 0)
        LOGR("diag: work-queue[%u] thread=%X object=%08X head=%08X count=%u limit=%u",
             n, kernel::Current() ? kernel::Current()->id : 0, object,
             object >= 0x10000 ? kernel::R32(object + 4) : 0,
             kernel::R32(0x005E5D80), kernel::R32(0x0056FC90));
    __imp__sub_00281270(ctx, base);
}

