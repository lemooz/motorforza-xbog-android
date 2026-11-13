// XAPI functions that talk to hardware: controller ports and the hard disk cache partition
#include <chrono>

#include "core/input.h"
#include "gpu/backend.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

using namespace kernel;

namespace {

// XPP device type records: CurrentConnected, ChangeConnected, PreviousConnected
constexpr uint32_t kGamepadType = SYM(XPP_GamepadType);
constexpr uint32_t kMemoryUnitType = SYM(XPP_MemoryUnitType);

std::mutex g_ports_lock;
bool g_poller = false;

void UpdatePorts() {
    std::lock_guard<std::mutex> lock(g_ports_lock);
    uint32_t cur = 0;
    for (int p = 0; p < 4; p++)
        if (input::Poll(p).connected) cur |= 1u << p;
    uint32_t old = R32(kGamepadType + XDK_XPP_Current);
    if (cur != old) {
        W32(kGamepadType + XDK_XPP_Changed, R32(kGamepadType + XDK_XPP_Changed) | (cur ^ old));
        W32(kGamepadType + XDK_XPP_Current, cur);
        LOGI("xapi: controller ports now %X", cur);
    }
}

struct Handle {
    uint32_t port;
    uint32_t packet;
    input::Pad last;
};
std::unordered_map<uint32_t, Handle> g_handles;

}  // namespace

HLE_STDCALL(XMountUtilityDrive, 1) {
    CreateSymbolicLink("\\??\\Z:", "\\Device\\Harddisk0\\Partition5");
    return 1;
}

HLE_STDCALL(XInitDevices, 2) {
    XDK_REQUIRE("xapi", XDK_KEY(XPP_Current), XDK_KEY(XPP_Changed), XDK_KEY(XPP_Previous), XDK_DATA(XPP_GamepadType));
    W32(kGamepadType + XDK_XPP_Current, 0);
    W32(kGamepadType + XDK_XPP_Changed, 0);
    W32(kGamepadType + XDK_XPP_Previous, 0);
    if constexpr (HAS_XPP_MemoryUnitType != 0) {  // titles without memory unit support have no record
        W32(kMemoryUnitType + XDK_XPP_Current, 0);
        W32(kMemoryUnitType + XDK_XPP_Changed, 0);
        W32(kMemoryUnitType + XDK_XPP_Previous, 0);
    }
    UpdatePorts();
    if (!g_poller) {
        g_poller = true;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                UpdatePorts();
            }
        }).detach();
    }
    return 0;
}

HLE_STDCALL(XInputOpen, 4) {
    static std::atomic<uint32_t> diag_open{0};
    uint32_t type = a[0], port = a[1];
    if (type != kGamepadType || port > 3) return 0;
    uint32_t h = mem::AllocSystem(16);
    std::lock_guard<std::mutex> lock(g_ports_lock);
    g_handles[h] = {port, 0, {}};
    uint32_t n = diag_open.fetch_add(1, std::memory_order_relaxed);
    if (n < 8) LOGR("diag: xinput-open[%u] type=%08X port=%u handle=%08X", n, type, port, h);
    return h;
}

HLE_STDCALL(XInputClose, 1) {
    std::lock_guard<std::mutex> lock(g_ports_lock);
    g_handles.erase(a[0]);
    mem::FreeSystem(a[0]);
    return 0;
}

HLE_STDCALL(XInputGetState, 2) {
    static std::atomic<uint32_t> diag_state{0};
    uint32_t diag_n = diag_state.fetch_add(1, std::memory_order_relaxed);
    const uint32_t caller = kernel::R32(ctx.esp);
    uint32_t port;
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        auto it = g_handles.find(a[0]);
        if (it == g_handles.end()) return 6;  // ERROR_INVALID_HANDLE
        port = it->second.port;
    }
    input::Pad p = input::Poll(int(port));
    if (diag_n < 32) LOGR("diag: xinput-state[%u] port=%u connected=%u A=%u buttons=%04X",
                          diag_n, port, p.connected, p.analog[0], p.buttons);
    if (diag_n % 3000 == 0)
        LOGR("diag: xinput-poll[%u] caller=%08X port=%u connected=%u A=%u buttons=%04X",
             diag_n, caller, port, p.connected, p.analog[0], p.buttons);
    if (!p.connected) return 1167;  // ERROR_DEVICE_NOT_CONNECTED
    std::lock_guard<std::mutex> lock(g_ports_lock);
    Handle& h = g_handles[a[0]];
    const bool changed = std::memcmp(&p, &h.last, sizeof(p)) != 0;
    if (changed) {
        h.packet++;
        LOGR("diag: xinput-change port=%u packet=%u caller=%08X connected=%u A=%u buttons=%04X lx=%d ly=%d",
             port, h.packet, caller, p.connected, p.analog[0], p.buttons, p.lx, p.ly);
        if (p.analog[0] || p.buttons) {
            gpu::g_capture_front_after_input.store(true, std::memory_order_release);
            gpu::g_capture_present_after_input.store(true, std::memory_order_release);
            gpu::g_frontend_draw_diag.store(1, std::memory_order_release);
            gpu::g_post_input_draw_diag.store(0, std::memory_order_release);
        }
        // R3 traces the next complete render frame. L3 captures a bounded run
        // of original Forza material draws without replacing their pixel shader.
        const uint16_t stick_press = p.buttons & ~h.last.buttons & 0x00C0;
        if (stick_press) {
            if (stick_press & 0x0040) {
                const uint32_t window = gpu::g_capture_indexed_sequence_window.fetch_add(1, std::memory_order_acq_rel);
                // The backend enumerates distinct material pairs in one guest
                // present. Advancing by 256 skipped every pair after the first
                // page because that per-present ordinal never reached 256.
                // Move one capture page (16 pairs) at a time so repeated L3
                // presses can walk from track materials to the player's car.
                gpu::g_capture_indexed_sequence_skip.store(window * 16, std::memory_order_release);
                gpu::g_capture_indexed_sequence.store(16, std::memory_order_release);
                gpu::g_capture_indexed_color.store(3, std::memory_order_release);
            }
            if (stick_press & 0x0080) gpu::g_capture_render_frame.store(true, std::memory_order_release);
            gpu::g_post_input_clear_diag.store(64, std::memory_order_release);
        }
    }
    h.last = p;
    uint32_t s = a[1];
    W32(s, h.packet);  // XINPUT_STATE: packet, buttons, 8 analog buttons, thumbs
    W16(s + 4, p.buttons);
    std::memcpy(HostPtr(s + 6), p.analog, 8);
    W16(s + 14, uint16_t(p.lx));
    W16(s + 16, uint16_t(p.ly));
    W16(s + 18, uint16_t(p.rx));
    W16(s + 20, uint16_t(p.ry));
    return 0;
}

HLE_STDCALL(XInputSetState, 2) {
    uint32_t port;
    {
        std::lock_guard<std::mutex> lock(g_ports_lock);
        auto it = g_handles.find(a[0]);
        if (it == g_handles.end()) return 6;
        port = it->second.port;
    }
    uint32_t fb = a[1];
    // XINPUT_FEEDBACK_HEADER is dwStatus, hEvent and 58 reserved bytes, the rumble follows
    input::Rumble(int(port), R16(fb + 66), R16(fb + 68));
    W32(fb, 0);
    if (uint32_t ev = R32(fb + 4))
        if (ObjectRef o = FromHandle(ev)) SetEvent(*o);
    return 997;  // ERROR_IO_PENDING, already complete
}

// A recompiled image launches through the XAPI, then a process restart into its executable; any other
// (the dashboard, updates) is refused and the title stays in its menu.
X86_EXTERN_FUNC(GUEST_ORIG(XLaunchNewImageA));
X86_FUNC(GUEST(XLaunchNewImageA)) {
    std::string path = ReadCString(R32(ctx.esp + 4));
    if (ImageExe(path)) {
        GUEST_ORIG(XLaunchNewImageA)(ctx, base);
        return;
    }
    LOGW("xapi: launch of %s refused, it is not recompiled", path.c_str());
    Return(ctx, 2, 2);  // ERROR_FILE_NOT_FOUND
}
