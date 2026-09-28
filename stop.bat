@echo off
chcp 65001 >nul
cd /d "%~dp0"
call stop_services.bat
