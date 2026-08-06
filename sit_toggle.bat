@echo off
rem Sit-key helper: presses the sit key every 10s via VIIPER (Ctrl+C to stop).
cd /d "%~dp0"
py -3 scripts/sit_toggle.py %*
echo.
pause
