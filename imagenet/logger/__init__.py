import os
from typing import Union


from .base import BaseLogger, make_output_format, get_wandb_expr_info
from .image_logger import ImageLogger
from .molecule_logger import MoleculeLogger
from .audio_logger import AudioLogger

# placeholder that will be replaced by the actual logger via setup_logger called by get_config()
logger = None

def log_samples(*args, **kwargs):
    assert logger is not None, "Logger is not initialized"
    logger.log_samples(*args, **kwargs)

def load_samples(*args, **kwargs):
    assert logger is not None, "Logger is not initialized"
    return logger.load_samples(*args, **kwargs)

def log_metrics(*args, **kwargs):
    assert logger is not None, "Logger is not initialized"
    logger.log_metrics(*args, **kwargs)

def log(*args, **kwargs):
    assert logger is not None, "Logger is not initialized"
    logger.log(*args, **kwargs)

def setup_logger(args) -> Union[BaseLogger, ImageLogger, MoleculeLogger, AudioLogger]:

    dir = os.path.expanduser(args.logging_dir)
    log_suffix = args.log_suffix

    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    format_strs = "stdout,log".split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
    
    if args.data_type == 'image':
        _logger = ImageLogger(args, output_formats)
    elif args.data_type == 'molecule':
        _logger = MoleculeLogger(args, output_formats)
    elif args.data_type == 'text2image':
        _logger = ImageLogger(args, output_formats)
    elif args.data_type == 'audio':
        _logger = AudioLogger(args, output_formats)
    else:
        _logger = BaseLogger(args, output_formats)

    global logger
    logger = _logger
    if output_formats:
        log("Logging to %s" % dir)
    
    log(f"Arguments: {args}")

    if args.wandb:
        log_metrics(get_wandb_expr_info(args))
    