# MotorForza (Forza Motorsport 2005) — Native Android Port

An unofficial native Android port of the original Xbox racing classic **Forza Motorsport** (2005), powered by static recompilation via [XboxXBE-Recomp](https://github.com/TGP482/XboxXBE-Recomp), native Vulkan 1.1+ rendering, SDL3 platform integration, and customized on-screen touch controls with gyroscope steering.

> [!IMPORTANT]
> **No game assets, executables, or copyrighted materials are distributed in this repository or with any release.**  
> Users must legally own an original Xbox copy of *Forza Motorsport* and supply their own `default.xbe` or disc image (`.iso`).

---

## Features

- **Native ARM64 Execution**: Ahead-of-time (AOT) static C++ recompilation. No full-system emulation overhead, running directly on Android hardware.
- **Vulkan Graphics Pipeline**: Modern Vulkan backend optimized for mobile tile-based deferred renderers (TBDR):
  - Coalesced render passes with `LOAD_OP_CLEAR` fast-paths.
  - Geometry and cubemap reflection caching by content serial.
  - Configurable dynamic resolution scaling and internal resolution scaling (480p up to native 1080p).
  - Anisotropic filtering, bloom toggle, anti-aliasing (FXAA / SSAA) and customizable screen post-filters.
  - Multi-GPU driver loader interface (supporting Qualcomm Adreno / Mesa Turnip, ARM Mali / PanVK, and Samsung Exynos Xclipse).
- **Control Options**:
  - Touch controls tailored for racing (analog gas/brake pedal sliders, virtual steering wheel, face buttons).
  - Gyroscope steering with adjustable deadzones and sensitivity.
  - Full physical gamepad support (Xbox, PlayStation, standard HID) with vibration feedback.
  - Keyboard & mouse support.
- **Audio & System**:
  - AAudio integration through SDL3 with automatic 5.1-to-stereo folding.
  - Background audio & pause support.

---

## Installation & How to Play

### 1. Download & Install the APK
Download the latest APK from the [Releases](https://github.com/lemooz/motorforza-xbog-android/releases) section and install it on your device.

### 2. Grant Storage Access
Launch **Forza Motorsport** and grant "All Files Access" (or Storage permission on Android 10) when prompted so the application can locate your game files and write save data.

### 3. Supply Your Game Files
Copy the files from your legally acquired Xbox copy to your phone:
- Path: `/storage/emulated/0/Forza/` (or a `Forza/` directory at the root of your MicroSD card).
- Supported formats:
  - An `.iso` disc image (e.g. `/storage/emulated/0/Forza/forza.iso`).
  - Or an extracted folder containing `default.xbe` along with its `media/` directory.

The launcher will detect your game files automatically. Tap **JOGAR** to start racing!

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Android 10 (API 29) | Android 13+ (API 33+) |
| **Architecture** | 64-bit ARM (arm64-v8a) | 64-bit ARM (arm64-v8a) with ARMv8.2-a+ |
| **GPU** | Vulkan 1.1 compatible (Adreno 6xx, Mali-G68 / G7x, Xclipse) | Adreno 7xx / Mali-G715 / Dimensity 8000+ |
| **RAM** | 4 GB | 6 GB or higher |
| **Storage** | ~8 GB free space for game files and shader cache | High-speed UFS storage |

---

## Building from Source

Building the project requires a Windows PC:

1. **Prerequisites**:
   - Visual Studio 2022 (with *Desktop development with C++*) and standalone LLVM/Clang (version 18+).
   - [Android Studio](https://developer.android.com/studio) with Android SDK Platform 34, NDK `28.2.13676358` (or 27+), and CMake `3.31.6`.
   - Python 3.10+ (for asset extraction).
2. **Setup**:
   - Clone this repository:
     ```bash
     git clone https://github.com/lemooz/motorforza-xbog-android.git
     cd motorforza-xbog-android
     ```
   - Place your legally dumped `default.xbe` into `assets/` (or your `.iso` into `iso/`).
3. **Compile**:
   - Run `build.bat` from a developer command prompt.
   - The batch script recompiles the x86 code into C++ (`generated/game`) and invokes Gradle to build the signed Android APK.

---

## Legal & Disclaimer

*Forza Motorsport* is a registered trademark of Microsoft Corporation. This project is an independent open-source research and preservation effort. It is not affiliated with, endorsed by, or associated with Microsoft, Turn 10 Studios, or Xbox. No proprietary source code, binaries, or assets from the original game are included in this repository.

---

## Credits & Acknowledgements

- [XboxXBE-Recomp](https://github.com/TGP482/XboxXBE-Recomp) by TGP482 and contributors.
- [SDL3](https://github.com/libsdl-org/SDL) for cross-platform Android windowing, audio, and controller handling.
- [Dear ImGui](https://github.com/ocornut/imgui) for in-game diagnostic and settings overlays.
- [glslang](https://github.com/KhronosGroup/glslang) for runtime SPIR-V shader compilation.
- The open-source Xbox preservation and ARM recompilation communities (Black-Android, blinx-android, hakuX).
