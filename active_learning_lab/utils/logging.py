import json
import logging
import loguru
import sys


class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger = loguru.logger
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class StreamToLogger(object):

    def __init__(self, logger, level='INFO'):
        self.logger = logger
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            self.logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


def setup_logger(output_file):
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    all_logger = logging.getLogger('active_learning_lab')
    all_logger.setLevel(logging.DEBUG)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)

    dynamo_logger = logging.getLogger('torch._dynamo')
    dynamo_logger.setLevel(logging.ERROR)

    # only for eacl23 until we can better control fit_kwargs / setfit logging
    import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    if sys.version.startswith('3.7'):
        logger = logging.getLogger()
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
        logging.basicConfig(handlers=[InterceptHandler()], level=0)
    else:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logger = loguru.logger

    fmt = '<green>{time:YY-MM-DD HH:mm:ss}</green> ' \
          '<level>{extra[shortlevel]}</level> ' \
          '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | ' \
          '<level>{message}</level>\n'

    def formatter(record):
        name = record['name']
        parts = name.split('.')
        record['name'] = '.'.join([part[0] for part in parts[:-1]] + [parts[-1]])

        record['extra']['shortlevel'] = record['level'].name[0]

        return fmt

    logger.remove()
    logger.add(sys.stderr, colorize=True, level='DEBUG', format=formatter)
    logger.add(output_file, level='DEBUG', format=formatter)

    return logger


def log_experiment_info(logger, active_run, name, sep_width=80):

    logger.info('*')
    logger.info('-' * sep_width)
    logger.info('Experiment: {} (ID: {})', name, active_run.info.experiment_id)
    logger.info('Active run: {}', active_run.info.run_id)
    logger.info('-' * sep_width)


def log_args(logger, args, title='Args:'):
    def default(o):
        return { 'unserializable_cls': str(type(o)) }

    logger.info(f'{title}\n{json.dumps(args, indent=4, default=default)}')
