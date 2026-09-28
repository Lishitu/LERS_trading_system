@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [LERS] 正在停止策略后台服务 (Web 看板与飞书守护)...
set "PY_CMD=python"
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist "D:\anaconda3\python.exe" (
        set "PY_CMD=D:\anaconda3\python.exe"
    )
)
%PY_CMD% -c "import psutil; targets=['web/app.py', 'cli.py daemon', 'feishu_bot.py']; [p.kill() for p in psutil.process_iter(['pid','cmdline']) if any(t in ' '.join(p.info.get('cmdline') or []).replace('\\','/') for t in targets)]" 2>nul
echo [LERS] 后台服务已安全停止。
timeout /t 2 /nobreak >nul
