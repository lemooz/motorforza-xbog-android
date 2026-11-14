#include <algorithm>
#include <iterator>

#include <toml++/toml.hpp>

#include "app/app.h"
#include "audio/output.h"
#include "common/menu.h"
#include "common/mouse_keyboard.h"
#include "common/options.h"

namespace game {

namespace {

// port 0 has the touch controls and, with a keyboard plugged in, its bindings; the game sees no input while the
// panel is open
void Pad(int port, input::Pad& pad) {
    input::SeenPad(pad);
    if (port == 0) {
        pad.connected = true;
        input::Keyboard(pad);  // the action bindings (WASD and so on) on top of the pad's own state
    }
    if (app::MenuOpen()) pad = input::Pad{.connected = pad.connected};
}

// no vibration while the panel is up
void Rumble(int port, uint16_t& left, uint16_t& right) {
    if (app::MenuOpen()) {
        left = 0;
        right = 0;
        return;
    }
    float scale = std::clamp(Opt().force_feedback, 0.0f, 1.0f);
    left = uint16_t(left * scale);
    right = uint16_t(right * scale);
}

// what the options list cannot hold
void Load(const void* table) {
    const toml::table& t = *static_cast<const toml::table*>(table);
    Options& o = Opt();
    o.mouse_sensitivity = float(t["mouse_sensitivity"].value_or(double(o.mouse_sensitivity)));
    o.touch_sensitivity = float(t["touch_sensitivity"].value_or(double(o.mouse_sensitivity)));
    o.left_sensitivity = float(t["left_stick_sensitivity"].value_or(double(o.left_sensitivity)));
    o.right_sensitivity = float(t["right_stick_sensitivity"].value_or(double(o.right_sensitivity)));
    o.force_feedback = std::clamp(float(t["force_feedback"].value_or(double(o.force_feedback))), 0.0f, 1.0f);
    o.gamepad_preset = std::min(uint32_t(t["gamepad_preset"].value_or(int64_t(o.gamepad_preset))), 2u);
    if (const toml::table* keys = t["keys"].as_table())
        for (auto&& [id, value] : *keys)
            o.keys[std::string(id.str())] = value.value_or(std::string());
}

// after the settings load, before boot
void Init() {
    BootLanguage();
    audio::SetOutputMode(Opt().audio_output);
    input::LoadBindings();
}

void Save(void* table) {
    toml::table& t = *static_cast<toml::table*>(table);
    const Options& o = Opt();
    t.insert("mouse_sensitivity", double(o.mouse_sensitivity));
    t.insert("touch_sensitivity", double(o.touch_sensitivity));
    t.insert("left_stick_sensitivity", double(o.left_sensitivity));
    t.insert("right_stick_sensitivity", double(o.right_sensitivity));
    t.insert("force_feedback", double(o.force_feedback));
    t.insert("gamepad_preset", int64_t(o.gamepad_preset));
    toml::table keys;
    for (const auto& [id, value] : o.keys) keys.insert(id, value);
    t.insert("keys", keys);
}

}  // namespace

Options& Opt() {
    static Options o;
    return o;
}

const std::string& BootLanguage() {
    static const std::string language = "int";
    return language;
}

bool LanguageAvailable(const std::string&) { return true; }

// Forza's HUD is game code the port has no hook into yet
bool HudShown() { return true; }

}  // namespace game

const app::GameInfo app::kGame = {
    .title = L"Forza Motorsport",
    .short_title = L"Forza Motorsport",
    .name = L"Forza",
    .log = "forza",
    .settings = "forza.toml",
    .splash = nullptr,
    .missing = L"Put the game in the Forza folder on the phone's storage: its disc image (.iso), or default.xbe with "
               L"the media folder beside it.",
    .images = {{"default.xbe", "Forza"}},
    .options = {{"skip_intros", &game::Opt().skip_intros, nullptr},
                {"fps_counter", &game::Opt().fps_counter, nullptr},
                {"touch_style", nullptr, &game::Opt().touch_style},
                {"bloom", &game::Opt().bloom, nullptr},
                {"fog", &game::Opt().fog, nullptr},
                {"anti_aliasing", nullptr, &game::Opt().anti_aliasing},
                {"hdr", &game::Opt().hdr, nullptr},
                {"fmv_scale", nullptr, &game::Opt().fmv_scale},
                {"screen_filter", nullptr, &game::Opt().screen_filter},
                {"left_stick_deadzone", nullptr, &game::Opt().left_deadzone},
                {"right_stick_deadzone", nullptr, &game::Opt().right_deadzone},
                {"dynamic_resolution", &game::Opt().dynamic_resolution, nullptr},
                {"dynamic_target_fps", nullptr, &game::Opt().dynamic_target_fps},
                {"dynamic_min_percent", nullptr, &game::Opt().dynamic_min_percent},
                {"dynamic_max_percent", nullptr, &game::Opt().dynamic_max_percent},
                {"dynamic_hold", nullptr, &game::Opt().dynamic_hold},
                {"audio_output", nullptr, &game::Opt().audio_output},
                {"gyro_steering", &game::Opt().gyro_steering, nullptr}},
    .load = game::Load,
    .save = game::Save,
    .init = game::Init,
    .loaded = nullptr,
    .window = nullptr,
    .message = nullptr,
    .ui = app::DrawUi,
    .pad = game::Pad,
    .rumble = game::Rumble,
};
