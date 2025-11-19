import logging
import os
class LoggerManager:
    """
    Singleton manager for project-wide logging.
    Supports a global logger or per-run log files.
    """
    _logger = None

    @staticmethod
    def get_logger(log_dir='.', name="run", level=logging.INFO):
        """
        Returns a singleton logger instance for the project.
        """
        if LoggerManager._logger is not None:
            return LoggerManager._logger
        
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid adding multiple handlers if logger already exists
        if not logger.hasHandlers():
            # File handler
            log_file = os.path.join(log_dir, f"{name}.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(ch)
        LoggerManager._logger = logger
        return logger

    @staticmethod
    def log_uncaught_exceptions(exctype, value, tb):
        """
        Logs uncaught exceptions to the error log.
        """
        if LoggerManager._logger is None:
            raise RuntimeError("Logger must be initialized before logging exceptions.")
        LoggerManager._logger.error("Unhandled exception", exc_info=(exctype, value, tb))