"""Logging utilities for federated learning experiments."""
import logging
import sys
from datetime import datetime
from typing import Optional
import os


def setup_logger(name: str = "federated_traffic",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Setup and configure logger.

    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "federated_traffic") -> logging.Logger:
    """Get existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Logger for tracking experiment progress and metrics."""

    def __init__(self, experiment_name: str, output_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = datetime.now()

        # Create output directory
        self.exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Setup file logger
        log_file = os.path.join(self.exp_dir, "experiment.log")
        self.logger = setup_logger(experiment_name, log_file)

        self.logger.info(f"Experiment '{experiment_name}' started at {self.start_time}")

    def log_config(self, config: dict):
        """Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # Save config to file
        import yaml
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_round(self, round_num: int, metrics: dict):
        """Log metrics for a training round.

        Args:
            round_num: Round number
            metrics: Dict of metrics
        """
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items())
        self.logger.info(f"Round {round_num}: {metrics_str}")

    def log_final(self, metrics: dict):
        """Log final experiment results.

        Args:
            metrics: Final metrics
        """
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Experiment completed in {elapsed}")
        self.logger.info("Final Results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")

    def save_results(self, results: dict, filename: str = "results.yaml"):
        """Save results to file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        import yaml
        path = os.path.join(self.exp_dir, filename)
        with open(path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        self.logger.info(f"Results saved to {path}")
