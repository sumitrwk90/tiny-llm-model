import logging
from logging.handlers import RotatingFileHandler
import torch
import torch.distributed as dist
import random
import time
import traceback
from datetime import datetime

MAX_LOG_SIZE = 5*1024*1024
BACKUP_COUNT = 3
# ===============================
# 1. Setup Logger
# ===============================
def setup_logger(log_file="training.log"):
    logger = logging.getLogger("LLMPretraining")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if run multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format logs
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()