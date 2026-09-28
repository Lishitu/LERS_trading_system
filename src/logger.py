"""LERS 统一日志系统模块。

提供多通道高可靠日志记录：
- 彩色/格式化控制台输出
- 循环滚动文件持久化 (logs/system.log, logs/trade.log)
- 线程安全内存缓冲队列 (用于 Web 控制台实时日志轮询与前端展示)
"""

import logging
import os
import sys
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

# 全局内存日志缓冲区（最多保留 1000 条最近日志）
_LOG_BUFFER = deque(maxlen=1000)


class MemoryLogHandler(logging.Handler):
    """用于 Web 控制台实时读取的内存日志处理器。"""

    def emit(self, record):
        try:
            msg = self.format(record)
            _LOG_BUFFER.append({
                'time': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'raw': msg,
            })
        except Exception:
            self.handleError(record)


def get_recent_logs(n: int = 200, level: Optional[str] = None) -> List[Dict[str, str]]:
    """获取内存缓冲区中最近的日志记录。"""
    logs = list(_LOG_BUFFER)
    if level:
        level_upper = level.upper()
        logs = [entry for entry in logs if entry['level'] == level_upper]
    return logs[-n:]


def setup_logger(
    name: str = "LERS",
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """初始化并配置统一 Logger 实例。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复绑定 Handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 控制台 Handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 2. 滚动文件 Handler (每个文件最大 10MB，保留 5 个备份)
    if file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name.lower()}_system.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 3. 内存缓冲 Handler (供 Web 界面展示)
    mem_handler = MemoryLogHandler()
    mem_handler.setLevel(level)
    mem_handler.setFormatter(formatter)
    logger.addHandler(mem_handler)

    return logger


# 默认系统主 Logger 实例
system_logger = setup_logger("LERS")
