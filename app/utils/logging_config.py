import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


class LoggerConfig:
    """
    Configures a consistent logging system across the project.
    Supports console + rotating file logging.
    """

    def __init__(self, name: str = "GuitarMap",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 max_bytes: int = 5 * 1024 * 1024,
                 backup_count: int = 5,
                 use_colors: bool = True) -> None:
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.use_colors = use_colors

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """
        attach console + file handlers.
        :return:
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self._get_console_formatter())

        file_handler = RotatingFileHandler(self.log_dir / f"{self.name.lower()}.log", maxBytes=self.max_bytes,
                                           backupCount=self.backup_count, encoding="utf-8")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self._get_file_formatter())

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _get_console_formatter(self) -> logging.Formatter:
        """
        Use colored logs for better readability.
        :param self:
        :return:
        """
        if not self.use_colors:
            return logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

        COLORS = {
            "DEBUG": "\033[94m",
            "INFO": "\033[92m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "CRITICAL": "\033[95m",
            "RESET": "\033[0m",
        }

        class ColorFormatter(logging.Formatter):
            def format(self, record):
                color = COLORS.get(record.levelname, COLORS["RESET"])
                message = super().format(record)
                return f"{color}{message}{COLORS['RESET']}"

        return ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")

    @staticmethod
    def _get_file_formatter() -> logging.Formatter:
        """
        Plain text file formatter.
        :param self:
        :return:
        """
        return logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

    def get_logger(self):
        """Return the configured logger."""
        return self.logger


def get_logger(name: str = "GuitarMap") -> logging.Logger:
    """Retrieve a global project logger."""
    config = LoggerConfig(name=name)
    return config.get_logger()
