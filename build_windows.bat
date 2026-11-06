@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Create a Windows virtual environment at .venv first.
    echo See WINDOWS_BUILD.md for the offline build procedure.
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed in .venv.
    echo Install it before running this script.
    exit /b 1
)

echo Building onedir distribution...
set RL_DEMO_ONEFILE=
python -m PyInstaller --clean --noconfirm RL_Demo.spec
if errorlevel 1 exit /b 1

echo.
echo Build complete: dist\RL_Demo\RL_Demo.exe
echo Run that executable for offline testing before creating onefile.
echo.
echo After onedir testing, use:
echo   set RL_DEMO_ONEFILE=1
echo   python -m PyInstaller --clean --noconfirm RL_Demo.spec
endlocal
