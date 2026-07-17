"""
统一日志配置

用法:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("数据加载完成，共 %d 行", len(df))

级别:
    DEBUG    - 调试细节（特征值、中间变量）
    INFO     - 常规信息（阶段完成、数量统计）
    WARNING  - 警告（数据不足、回退默认值）
    ERROR    - 错误（但可恢复）
    CRITICAL - 严重错误（程序无法继续）
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    创建或获取一个 logger。

    参数:
        name: logger 名称，通常传 __name__（会自动显示模块名）
        level: 日志级别（logging.DEBUG / INFO / WARNING / ERROR）
        log_dir: 日志目录，默认 <项目根>/logs/

    返回:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（同一个 logger 多次调用 setup_logger）
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── 确定日志目录 ──
    if log_dir is None:
        # 自动检测项目根目录（当前文件所在目录的父目录）
        log_dir = str(Path(__file__).parent / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # ── Handler 1: 文件日志（永久保存，完整格式） ──
    # 按模块名分文件，方便查找
    module_name = name.replace(".", "_")
    log_file = Path(log_dir) / f"{module_name}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # ── Handler 2: 控制台日志（实时看，简洁格式） ──
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(levelname)-8s | %(message)s"
    ))
    logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    快速获取已配置的 logger。

    第一次调用 __main__ 模块时自动初始化，
    之后其他模块通过 get_logger(__name__) 获取同名的配置。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果还没配置，用默认设置初始化
        return setup_logger(name)
    return logger
