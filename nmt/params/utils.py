"""
Miscellaneous
"""
import os
import os.path as osp
import collections
import logging


def read_list(param):
    """ Parse list of integers """
    param = str(param)
    param = [int(p) for p in param.split(',')]
    return param


def update(d, u):
    """update dict of dicts"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_densenet_params(args):
    if "network" in args:
        if "num_layers" in args['network']:
            num_layers = read_list(args['network']['num_layers'])
            if "kernels" in args['network']:
                kernels = read_list(args['network']['kernels'])
                if len(kernels) == 1:
                    args['network']['kernels'] = kernels * len(num_layers)
                else:
                    assert len(kernels) == len(num_layers), "the number of kernel sizes must match that of the network layers"
                    args["network"]["kernels"] = kernels
            args['network']['num_layers'] = num_layers
    return args


class ColorStreamHandler(logging.StreamHandler):
    """Logging with colors"""
    DEFAULT = '\x1b[0m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    CYAN = '\x1b[36m'

    CRITICAL = RED
    ERROR = RED
    WARNING = YELLOW
    INFO = GREEN
    DEBUG = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:
            return cls.CRITICAL
        if level >= logging.ERROR:
            return cls.ERROR
        if level >= logging.WARNING:
            return cls.WARNING
        if level >= logging.INFO:
            return cls.INFO
        if level >= logging.DEBUG:
            return cls.DEBUG
        return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return color + text + self.DEFAULT


def create_logger(job_name, log_file=None, debug=True):
    """
    Initialize global logger
    log_file: log to this file, besides console output
    return: created logger
    """
    logging.basicConfig(level=5,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.root.handlers = []
    if debug:
        chosen_level = 5
    else:
        chosen_level = logging.INFO
    logger = logging.getLogger(job_name)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%m/%d %H:%M')
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
        # cerate file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(chosen_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Colored stream handler
    sh = ColorStreamHandler()
    sh.setLevel(chosen_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
