import logging
from arguments import args


# def _setup_logger():
#     log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(log_format)
#     logger.handlers = [console_handler]

#     return logger


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler

    if args.task == 'train':
        log_file = 'train.log'
    elif args.task == 'test':
        log_file = 'test.log'
    elif args.task == 'eval':
        log_file = 'eval.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)


    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = _setup_logger()