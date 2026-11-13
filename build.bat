@echo off
setlocal
set "GAME_TITLE=Forza Motorsport"
set "GAME_XBE=default.xbe"
title %GAME_TITLE% - Build
cd /d "%~dp0"
echo.
echo   %GAME_TITLE%
echo   =================
echo.

set "PF86=%ProgramFiles(x86)%"
set "VSWHERE=%PF86%\Microsoft Visual Studio\Installer\vswhere.exe"
set "TOOLS=%CD%\out\build\android-tools"
if not defined ANDROID_HOME set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
set "NDK=%ANDROID_HOME%\ndk\28.2.13676358"
set "SDKCMAKE=%ANDROID_HOME%\cmake\3.31.6\bin"

rem ---------------------------------------------------------------- tools
echo [1/5] Tools
if not exist "%ANDROID_HOME%\platforms" goto :nosdk
if not exist "%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\clang-cl.exe" goto :nosdk
if not exist "%SDKCMAKE%\cmake.exe" goto :nosdk
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo       Visual Studio with "Desktop development with C++" was not found: the recompiler links with its tools.
    goto :fail
)
call "%VSDIR%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >nul
set "PATH=%SDKCMAKE%;%PATH%"
rem The recompiler for Windows: standalone LLVM (llvm.org) when installed, else Visual Studio's Clang component,
rem else the NDK's Clang. New MSVC headers need Clang 20 or newer, so the NDK's only works with older Visual Studio
set "CLANGCL=C:\Program Files\LLVM\bin\clang-cl.exe"
if not exist "%CLANGCL%" set "CLANGCL=%ProgramFiles(x86)%\LLVM\bin\clang-cl.exe"
if not exist "%CLANGCL%" set "CLANGCL=%VSDIR%\VC\Tools\Llvm\x64\bin\clang-cl.exe"
if not exist "%CLANGCL%" set "CLANGCL=%NDK%\toolchains\llvm\prebuilt\windows-x86_64\bin\clang-cl.exe"
if not exist "%CLANGCL%" (
    echo       clang-cl was not found: install LLVM from https://llvm.org/builds/ or the Clang component in Visual Studio.
    goto :fail
)
where git >nul 2>nul || (echo       git is missing: https://git-scm.com/download/win & goto :fail)
rem Java 17 or newer to run Gradle: Android Studio's JBR first, else a JAVA_HOME that has a compiler. A JAVA_HOME
rem left on an old JRE (Java 8) would make Gradle fail confusingly, so it is only kept when it can compile
set "JDK="
for %%j in ("%ProgramFiles%\Android\Android Studio\jbr" "%ProgramFiles%\Android\Android Studio1\jbr" "%LOCALAPPDATA%\Programs\Android Studio\jbr") do (
    if exist "%%~j\bin\javac.exe" if not defined JDK set "JDK=%%~j"
)
if not defined JDK if defined JAVA_HOME if exist "%JAVA_HOME%\bin\javac.exe" set "JDK=%JAVA_HOME%"
if not defined JDK (
    echo       No Java 17 or newer: install Android Studio, or set JAVA_HOME to a JDK.
    goto :fail
)
set "JAVA_HOME=%JDK%"

rem ---------------------------------------------------------------- the xbe
echo [2/5] %GAME_XBE%
if exist assets\%GAME_XBE% goto :havexbe
set "ISO="
for %%f in (*.iso assets\*.iso iso\*.iso) do if not defined ISO set "ISO=%%f"
if not defined ISO goto :nogame
where python >nul 2>nul || (echo       %GAME_XBE% is not in assets and Python is missing to take it from %ISO%. & goto :nogame)
if not exist assets mkdir assets
echo       taking it from %ISO%
python -I tools\xdvdfs.py "%ISO%" %GAME_XBE% assets\%GAME_XBE% || goto :fail
:havexbe

rem ---------------------------------------------------------------- recompiler
echo [3/5] Recompiler
cmake -S tools -B "%TOOLS%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=%CLANGCL%" "-DCMAKE_CXX_COMPILER=%CLANGCL%" >nul || goto :fail
cmake --build "%TOOLS%" >nul || goto :fail
set /p SDK=<"%TOOLS%\sdk_dir.txt"
set "SDK=%SDK:/=\%"
findstr /b /l /c:"[symbols]" config\game.toml >nul 2>nul && goto :recompile
"%TOOLS%\xbe_symbols.exe" assets\%GAME_XBE% "%SDK%\xdk" --out config\game.toml || goto :fail

:recompile
echo [4/5] Recompiling %GAME_XBE% into generated\game
"%TOOLS%\xbe_recomp.exe" config\game.toml --xdk "%SDK%\xdk" >nul || goto :fail

rem ---------------------------------------------------------------- Android
echo [5/5] APK: the game code for arm64, 30 to 90 minutes the first time
if not exist local.properties echo sdk.dir=%ANDROID_HOME:\=/%> local.properties
rem GRADLE_ARGS: more for Gradle, as the release workflow's -PversionCode and -PversionName
call gradlew.bat assembleRelease %GRADLE_ARGS% || goto :fail

echo.
echo   %GAME_TITLE% for Android built.
echo   app\build\outputs\apk\release\app-release.apk
echo.
echo   Install it with: adb install -r app\build\outputs\apk\release\app-release.apk
echo   Then copy the game files to the phone, see README.md.
echo.
if not defined CI pause
exit /b 0

:nosdk
echo       The Android SDK was not found complete. Install Android Studio, then in its SDK Manager:
echo       Android SDK Platform 34, NDK 28.2.13676358 and CMake 3.31.6.
goto :fail

:nogame
echo   Put the game in the assets folder first: %GAME_XBE% from the USA release, or its disc image (.iso)
echo     %CD%\assets
goto :fail

:fail
echo.
echo   BUILD FAILED. Read the last message above.
echo.
if not defined CI pause
exit /b 1
