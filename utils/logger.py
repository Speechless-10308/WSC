import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="./logs", log_file="training.log", log_level=logging.INFO):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

    def log_args(self, args):
        original_formatters = []
        for handler in self.logger.handlers:
            original_formatters.append(handler.formatter)
        simple_formatter = logging.Formatter("%(message)s")
        for handler in self.logger.handlers:
            handler.setFormatter(simple_formatter)
        self.logger.info("========== Running Configuration ==========")
        for key, value in vars(args).items():
            self.logger.info(f"{key:>20}: {value}")
        self.logger.info("===========================================")
        for handler, formatter in zip(self.logger.handlers, original_formatters):
            handler.setFormatter(formatter)