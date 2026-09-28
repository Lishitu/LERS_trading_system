@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 确保日志目录存在
if not exist "logs" mkdir "logs"

:: 检查 Python 运行程序
set "PY_CMD=pythonw"
where pythonw >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "D:\anaconda3\pythonw.exe" (
        set "PY_CMD=D:\anaconda3\pythonw.exe"
    ) else (
        set "PY_CMD=python"
    )
)

:: 1. 启动 Flask Web 界面端 (后台静默运行)
start "" %PY_CMD% web\app.py

:: 2. 启动飞书 WebSocket 长连接交互服务 (后台静默运行)
start "" %PY_CMD% cli.py daemon

:: 3. 稍候 2 秒确保端口就绪
timeout /t 2 /nobreak >nul

:: 4. 自动唤起默认浏览器访问 Web 看板
start http://127.0.0.1:5000
