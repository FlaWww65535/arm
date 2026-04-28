import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path


EXPERIMENT_LOG_CONFIG = {
    "log_root": "results/logs",
    "experiment_dir_format": "{dataset}_{embedding_model}_{lm}",
    "run_timestamp_format": "%Y-%m-%d_%H-%M-%S",
    "log_filename_format": "{run_timestamp}.log",
    "rotation_max_bytes": 500 * 1024,
    "rotation_backup_count": 1000,
}


class ExperimentLogger:
    @staticmethod
    def configure(stage: str, dataset: str, embedding_model: str, lm: str):
        experiment_dir_name = EXPERIMENT_LOG_CONFIG["experiment_dir_format"].format(
            stage=stage,
            dataset=dataset,
            embedding_model=embedding_model,
            lm=lm,
        )
        run_timestamp = datetime.now().strftime(
            EXPERIMENT_LOG_CONFIG["run_timestamp_format"]
        )
        log_dir = Path(EXPERIMENT_LOG_CONFIG["log_root"]) / experiment_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = EXPERIMENT_LOG_CONFIG["log_filename_format"].format(
            run_timestamp=run_timestamp,
        )
        log_path = log_dir / log_filename

        logger = logging.getLogger(stage)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        for handler in list(logger.handlers):
            if getattr(handler, "_arm_experiment_handler", False):
                logger.removeHandler(handler)
                handler.close()

        handler = RotatingFileHandler(
            log_path,
            maxBytes=EXPERIMENT_LOG_CONFIG["rotation_max_bytes"],
            backupCount=EXPERIMENT_LOG_CONFIG["rotation_backup_count"],
            encoding="utf-8",
        )
        setattr(handler, "_arm_experiment_handler", True)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s - [{stage}] - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)

        return logger
