import logging
import os
import sys
import datetime
import hashlib
import random
import platform
import numpy as np
import torch
import pandas as pd

GLOBAL_SEED = 42
EXPERIMENT_NAME = "CXR_Synapse_Foundation_ChestMNIST"
EXPERIMENT_ID = hashlib.sha256(f"{EXPERIMENT_NAME}_{GLOBAL_SEED}_{datetime.datetime.now(datetime.timezone.utc).isoformat()}".encode()).hexdigest()[:12]

def _configure_logger(name="cxr_synapse", level=None):
    """Create a single process-wide logger with consistent formatting."""
    logger = logging.getLogger(name)
    level_name = (level or os.getenv("CXR_LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    if logger.handlers:
        logger.setLevel(level_value)
        for handler in logger.handlers:
            handler.setLevel(level_value)
        return logger

    logger.setLevel(level_value)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level_value)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    return logger

LOGGER = _configure_logger()

def log_process(stage, event, level=logging.INFO, **details):
    """Structured process-change logging for traceability and maintainability."""
    if details:
        detail_str = ", ".join(f"{k}={details[k]}" for k in sorted(details))
        LOGGER.log(level, "[%s] %s | %s", stage, event, detail_str)
    else:
        LOGGER.log(level, "[%s] %s", stage, event)

def enforce_reproducibility(seed: int = GLOBAL_SEED) -> torch.Generator:
    """Lock all 6 sources of non-determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    g = torch.Generator()
    g.manual_seed(seed)
    return g

DATALOADER_GENERATOR = enforce_reproducibility(GLOBAL_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
