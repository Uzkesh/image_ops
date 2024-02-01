import logging


def init_logger(name: str):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stream_log = logging.StreamHandler()
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger
