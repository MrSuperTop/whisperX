import logging
import logging.config
from pathlib import Path

DEFAULT_MAIN_LOGGER_NAME = "whisperx"
current_main_logger_name: None | str = None


def setup_loggers(
    main_logger_name: str = DEFAULT_MAIN_LOGGER_NAME, log_file: Path = Path("./runtime.log")
) -> logging.Logger:
    global current_main_logger_name

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    logger = root.getChild(main_logger_name)
    current_main_logger_name = main_logger_name

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")

    file_handler.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    stdout_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # * Also adding the faster_whisper logger as DEBUG
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(current_main_logger_name).getChild(module_name)

    return logger
