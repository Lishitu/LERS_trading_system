@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   [LERS] 正在后台启动量化策略服务并唤起 Web 界面端...
echo ========================================================
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python launcher.py
) else if exist "D:\anaconda3\python.exe" (
    "D:\anaconda3\python.exe" launcher.py
) else (
    echo [ERROR] 未找到 Python 环境，请安装 Python 并添加至系统 PATH。
    pause
)
