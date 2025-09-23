import logging
import logging.config
import logging.handlers
import sys

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(filename)s:%(funcName)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'console': {
            'format': '%(message)s'
        },
    },
    'handlers': {
        'fileHandler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'logs.log',
            'mode': 'a',
            'maxBytes': 1024*1024*3,
        },
        'consoleHandler': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'console',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        'root': {
            'level': 'DEBUG',
            'handlers': ['fileHandler', 'consoleHandler']
        },
        'dynamic': {
            'level': 'INFO',
            'handlers': ['fileHandler', 'consoleHandler'],
            'propagate': False,
            'qualname': 'dynamic'
        }
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

def log_exceptions(func):
    import functools
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception in {self.__class__.__name__}.{func.__name__}")
            logger.error(f"Constructor arguments: {self.__dict__}")
            logger.error(f"Method arguments: {args}")
            logger.error(f"Method keyword arguments: {kwargs}")
            logger.error(f"Exception details: {str(e)}")
            # logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
    return wrapper

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = logging.getLogger(__name__)
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Initialize logging when this module is imported
setup_logging()
logger = logging.getLogger(__name__)