import logging
import sys
from config import LOG_LEVEL

def setup_logger(name: str = __name__) -> logging.Logger:
    """设置统一的日志记录器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # 避免重复添加处理器
        # 设置日志级别
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

# 创建全局日志记录器
logger = setup_logger("video_subtitle")
