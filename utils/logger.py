import os
import logging
from datetime import datetime

def setup_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Sets up a logger that outputs to both the console and a file.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory where log files should be stored.
        
    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicated output in notebooks or subsequent runs
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
