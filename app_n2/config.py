import os

# 处理参数配置
CHUNK_LENGTH_MS = int(os.getenv("CHUNK_LENGTH_MS", "15000"))  # ASR分块长度（毫秒）
TRANSLATION_CHUNK_LINES = int(os.getenv("TRANSLATION_CHUNK_LINES", "100"))  # 翻译分块行数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # 最大重试次数
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # 重试延迟（秒）

# 目录配置
TEMP_DIR = os.getenv("TEMP_DIR", "data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 前端默认配置值（用于界面初始化）
DEFAULT_ASR_BASE_URL = "http://localhost:7863"
DEFAULT_OPENAI_BASE_URL = "http://localhost:30000/v1"
DEFAULT_OPENAI_API_KEY = "your-api-key"
DEFAULT_OPENAI_MODEL = "Qwen2.5-7b"

# 兼容旧配置的环境变量读取（可选）
# 如果环境变量中有配置，可以作为默认值使用
if os.getenv("ASR_BASE_URL"):
    DEFAULT_ASR_BASE_URL = os.getenv("ASR_BASE_URL")
if os.getenv("OPENAI_BASE_URL"):
    DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if os.getenv("OPENAI_API_KEY"):
    DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if os.getenv("OPENAI_MODEL"):
    DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL")