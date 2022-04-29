import sys
from loguru import logger

from abyss.config.config_manager import ConfigManager

logger.remove()  # fresh start
logger.add(sys.stderr, level='INFO')
