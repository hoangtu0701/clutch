import os
import shutil
import stat

RELEASE_DIR = "CLUTCH_APP"
MODELS_DIR = os.path.join(RELEASE_DIR, "models")
LIBS_DIR = os.path.join(RELEASE_DIR, "libs")
SITE_PACKAGES = os.path.join("venv", "Lib", "site-packages")
EMBEDDED_PYTHON_SRC = "embedded_python"
EMBEDDED_PYTHON_DEST = os.path.join(RELEASE_DIR, "embedded_python")

def force_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Reset output dir
if os.path.exists(RELEASE_DIR):
    shutil.rmtree(RELEASE_DIR, onerror=force_remove_readonly)
os.makedirs(RELEASE_DIR)

# Copy python files
shutil.copy("clutch_loop.py", os.path.join(RELEASE_DIR, "clutch_loop.py"))

# Copy GSI config file
shutil.copy("gamestate_integration_clutch.cfg", os.path.join(RELEASE_DIR, "gamestate_integration_clutch.cfg"))

# Copy models folder
shutil.copytree("models", MODELS_DIR)

# Copy embedded Python folder
shutil.copytree(EMBEDDED_PYTHON_SRC, EMBEDDED_PYTHON_DEST)

# Copy site-packages to libs
shutil.copytree(SITE_PACKAGES, LIBS_DIR)

print("✅ Build complete. Check CLUTCH_APP/")

#testing

# Auto-generate run_loop.bat
loop_bat_path = os.path.join(RELEASE_DIR, "run_loop.bat")
with open(loop_bat_path, "w") as f:
    f.write(r"""@echo off
setlocal

REM Kill any leftover Python processes from last run
taskkill /F /IM python.exe >nul 2>&1

REM Move to script directory
cd /d %~dp0

REM Debug: print working dir
echo Running from: %cd%

REM Path to embedded Python
set PYTHON_DIR=embedded_python

REM Launch clutch_gsi.py
start "" "%PYTHON_DIR%\python.exe" clutch_loop.py

endlocal
exit
""")
    
print("✅ Created run_loop.bat")




