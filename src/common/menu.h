#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// the settings panel, drawn by DrawUi (GameInfo::ui); Forza has no options menu of its own to hold it
namespace app {

bool MenuOpen();
bool Capturing();  // a key binding waits for a key; no keyboard binds here, always false
void ToggleMenu();
void DrawUi(double fps);
inline constexpr uint32_t kFpsCapRefresh = 1;  // Settings::fps_cap: just under the display's refresh rate
void PaceFrame();  // app::Pace, with kFpsCapRefresh; game thread, after the present

// the panel's rows, in pages
namespace rows {

enum class Kind { Heading, Value };
struct Row {
    Kind kind;
    int id;
    const char* label;
};
// 0 graphics, 1 touch controls, 2 general, 3 credits
const Row* Page(int page, size_t* count);
std::string Value(const Row& row, int step = 0);  // what the row shows, stepped first when step is not zero

}  // namespace rows

}  // namespace app
