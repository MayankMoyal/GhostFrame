@echo off
echo ========================================================
echo Launching OBS Studio for Ghost Frame
echo (Enabling Microphone Access for Custom Docks)
echo ========================================================
echo.

set OBS_PATH="C:\Program Files\obs-studio\bin\64bit\obs64.exe"
set OBS_DIR="C:\Program Files\obs-studio\bin\64bit"

if exist %OBS_PATH% (
    echo Starting OBS Studio...
    start /d %OBS_DIR% obs64.exe --enable-media-stream
) else (
    echo OBS Studio not found at %OBS_PATH%.
    echo If you installed OBS somewhere else, please modify this file.
    pause
)
