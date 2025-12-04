import logging
import os
import wandb
import pandas as pd
import torch

class LoggerManager:
    """
    Singleton manager for project-wide logging.
    Supports a global logger or per-run log files.
    """
    _logger = None
    _cloud = False

    @staticmethod
    def get_logger(log_dir='.', name="run", level=logging.INFO, wandb_project=None, config=None):
        """
        Returns a singleton logger instance for the project.
        """
        # Return existing logger if already created
        if LoggerManager._logger is not None:
            return LoggerManager._logger
        
        # Create logger
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

        if wandb_project and not LoggerManager._cloud:
            # Additional cloud logging setup can be added here
            wandb_login()
            wandb.init(project=wandb_project, config=config, name=name)
            LoggerManager._cloud = True

        def send_file(title, path, type):
            '''
            Log images to wandb
            Example:
                logger.send_file("sample_image", "path/to/image.png", "image")
            Supported types: "image", "csv"
            '''
            if LoggerManager._cloud:
                if type == "image":
                    wandb.log({title: wandb.Image(path)})
                elif type == "csv":
                    artifact = wandb.Artifact(title, type="dataset")
                    artifact.add_file(path)
                    wandb.log_artifact(artifact)
                elif type == "table":
                    table = wandb.Table(dataframe=pd.read_csv(path))
                    wandb.log({title: table})
                else:
                    logger.error(f"Unsupported file type for wandb logging: {type}")
        
        def finish():
            '''
            Finish the wandb run
            '''
            logger.info("Finishing log...")
            if LoggerManager._cloud:
                wandb.finish()

        def log(message: str, data: bool = False):
            '''
            Unified log method
            - If data=False, logs plain text message
            - If data=True, logs text and parse key=vale pairs for wandb
            example for data=True:
                logger.log("epoch=1 loss=0.345 acc=0.89", data=True)
            '''
            logger.info(message)

            if data and LoggerManager._cloud:
                json_data = {}
                try:
                    parts = message.strip().split()
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            json_data[key] = float(value)
                except Exception as e:
                    logger.error(f"Failed to parse log message for cloud logging: {e}")
                    return
                wandb.log(json_data)
        # add method to logger
        logger.log = log
        logger.send_file = send_file
        logger.finish = finish
        LoggerManager._logger = logger
        logger.info("===== Start log =====")
        if config:
            logger.info(f"Config: {config}")
        return logger

def wandb_login():
    """
    Logs into W&B using an API key from environment variable 'WANDB_API_KEY'.
    Handles errors if the key is missing or login fails.
    """
    api_key = os.getenv("WANDB_API_KEY")
    
    if api_key is None:
        raise ValueError("Environment variable 'WANDB_API_KEY' not found. Please set it first.")
    
    try:
        # login() returns True if successful, False if already logged in
        logged_in = wandb.login(key=api_key)
        if logged_in:
            print("Logged into W&B successfully.")
        else:
            print("Already logged into W&B.")
    except wandb.errors.CommError as e:
        print(f"Failed to log into W&B: {e}")
    except Exception as e:
        print(f"Unexpected error during W&B login: {e}")
        # Add the log_tensor_state method to the logger instance
        def log_tensor_state(name: str, tensor: torch.Tensor):
            """
            Logs the state of a tensor, including whether it is complex and conjugated.

            Args:
                name (str): Name of the tensor.
                tensor (torch.Tensor): The tensor to log.
            """
            logger.info(f"Tensor {name}: is_complex={torch.is_complex(tensor)}, is_conj={tensor.is_conj()}")

        logger.log_tensor_state = log_tensor_state

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
