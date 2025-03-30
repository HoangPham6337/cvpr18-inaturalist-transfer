import os
from datetime import datetime
from typing import Optional


LOG_FILE_PATH: Optional[str] = None


def initialize_logger(log_dir: str = "./logs", filename: Optional[str] = None):
    global LOG_FILE_PATH
    os.makedirs(log_dir, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"log_{timestamp}.txt"
    LOG_FILE_PATH = os.path.join(log_dir, filename)


def log(message: str, verbose: bool = True, level: str = "INFO"):
    formatted = f"[{level}] {message}"
    if verbose:
        print(formatted)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{formatted}\n")
