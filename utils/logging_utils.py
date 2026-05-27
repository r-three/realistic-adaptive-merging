import logging
import os


def setup_logging() -> logging.Logger:
    """Set up logging configuration for the experiment.

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    if os.environ.get("SLURM_JOB_ID") is not None:
        logger.info(f"Slurm job id: {os.environ.get('SLURM_JOB_ID')}")

    return logger
