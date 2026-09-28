# -*- coding: utf-8 -*-
"""LERS 核心-卫星量化策略系统 · 一键后台启动器 (Zero-Console Window / Detached Daemon)."""

import os
import sys
import time
import socket
import subprocess
import webbrowser

base_dir = os.path.dirname(os.path.abspath(__file__))
python_exe = sys.executable

logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)


def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(port: int, max_wait: float = 15.0) -> bool:
    start_t = time.time()
    while time.time() - start_t < max_wait:
        if is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def main():
    # Windows 进程完全脱钩标记：独立运行、无黑窗、父进程退出不影响子进程
    DETACHED_FLAGS = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

    # 1. 检查并启动 Flask Web 界面端 (完全后台静默运行)
    if not is_port_open(5000):
        web_log = open(os.path.join(logs_dir, "web_server.log"), "a", encoding="utf-8")
        subprocess.Popen(
            [python_exe, os.path.join(base_dir, "web", "app.py")],
            cwd=base_dir,
            stdin=subprocess.DEVNULL,
            stdout=web_log,
            stderr=web_log,
            creationflags=DETACHED_FLAGS,
            close_fds=False,
        )

    # 2. 启动飞书 WebSocket 长连接守护进程 (完全后台静默运行)
    daemon_log = open(os.path.join(logs_dir, "daemon.log"), "a", encoding="utf-8")
    subprocess.Popen(
        [python_exe, os.path.join(base_dir, "cli.py"), "daemon"],
        cwd=base_dir,
        stdin=subprocess.DEVNULL,
        stdout=daemon_log,
        stderr=daemon_log,
        creationflags=DETACHED_FLAGS,
        close_fds=False,
    )

    # 3. 等待 Web 端口 5000 就绪后自动打开浏览器
    wait_for_port(5000, max_wait=15.0)

    # 4. 自动唤起系统默认浏览器访问前端界面
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
