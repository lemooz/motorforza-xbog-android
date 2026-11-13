#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <string>

// the port's own settings, in the settings file through app::kGame.options
namespace game {

// the languages the port's own text has; the game files carry the game's languages themselves
inline constexpr const char* kLanguages[] = {"int"};

struct Options {
    bool skip_intros = false;  // skip Bink intro videos at boot
    bool fps_counter = true;
    uint32_t fmv_scale = 0;  // 4:3 films on a wider screen: 0 fill it, top and bottom cut; 1 kept 4:3
    // a look over the frame: 0 none, 1 and 2 a CRT (light, strong), 3 sharpen, 4 vibrant, 5 noir, 6 sepia,
    // 7 night vision, 8 VHS
    uint32_t screen_filter = 0;
    bool bloom = false;  // the glow added over bright parts of the frame (native/vulkan.cpp)
    bool fog = true;    // the console's fog, which the renderer translates
    uint32_t anti_aliasing = 0;  // 0 off, 1 FXAA as the frame goes onto the screen, 2 SSAA: twice the resolution
    bool hdr = false;  // a tone map that brings back the colours the console's overbright look burns out
    // the render resolution follows the frame rate (native/dynamic_resolution.cpp): down when it falls under the
    // target, up again once it holds the target this many seconds; within min and max, percent of the screen's
    bool dynamic_resolution = false;
    uint32_t dynamic_target_fps = 60;
    uint32_t dynamic_min_percent = 40;
    uint32_t dynamic_max_percent = 100;
    uint32_t dynamic_hold = 5;
    float mouse_sensitivity = 1.0f;  // a mouse's look, and the gyro's steering
    float touch_sensitivity = 1.0f;  // the touch screen's own
    // a controller's sticks: the radial dead zone as percent of the travel, 0 the game's own, and what the travel
    // past it is scaled by; the left stick steers, the right one the gear menu and look
    uint32_t left_deadzone = 0;
    uint32_t right_deadzone = 0;
    float left_sensitivity = 1.0f;
    float right_sensitivity = 1.0f;
    float force_feedback = 1.0f;  // scale the controller's vibration
    uint32_t gamepad_preset = 0;  // input.cpp: standard, swap A/B, or swap sticks
    uint32_t audio_output = 0;  // audio::kOutputAuto, kOutputStereo, kOutputSurround or kOutputMono
    bool gyro_steering = false;  // the gyroscope turns the wheel (native/touch.cpp)
    std::map<std::string, std::string> keys;  // action id to key names, absent is the default
};
Options& Opt();
// the language setting as this run started
const std::string& BootLanguage();
bool LanguageAvailable(const std::string& code);  // translations.cpp: only the port's own text
inline std::atomic<float> g_fps{0};  // from the host, for a counter over the game's own
inline std::atomic<bool> g_hide_hud{false};
// the view's zoom as the look update scales the turn by: 1 none; kept for the touch controls' own logic
inline std::atomic<float> g_zoom{1.0f};
inline std::atomic<bool> g_focus_paused{false};  // the game thread waits while the app is in the background
inline std::atomic<uint32_t> g_focus_pauses{0};  // focus pauses ended so far
void WaitWhileUnfocused();  // native/game_host.cpp: game thread, before the present
bool HudShown();  // the game's HUD drew lately; true for Forza, which has no hook to tell yet

}  // namespace game
