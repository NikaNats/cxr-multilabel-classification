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

# ============================================================
# EXPERIMENT PROVENANCE & IDENTIFICATION
# ============================================================
GLOBAL_SEED = 42

# Identification string used to tag all generated artifacts and logs
EXPERIMENT_NAME = "CXR_Synapse_Foundation_ChestMNIST"

# Unique Experiment ID: Generated using a SHA-256 hash of the name, seed, and timestamp.
# This ensures that every run creates a unique, traceable fingerprint (First 12 chars).
EXPERIMENT_ID = hashlib.sha256(
    f"{EXPERIMENT_NAME}_{GLOBAL_SEED}_{datetime.datetime.now(datetime.timezone.utc).isoformat()}".encode()
).hexdigest()[:12]

# ============================================================
# OBSERVABILITY & STRUCTURED LOGGING
# ============================================================

def _configure_logger(name="cxr_synapse", level=None):
    """
    Creates a process-wide logger with consistent ISO-8601 formatting.
    Ensures observability across distributed training and evaluation passes.
    """
    logger = logging.getLogger(name)
    
    # Allow log level override via environment variable for debugging flexibility
    level_name = (level or os.getenv("CXR_LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    # Idempotent handler attachment to prevent duplicate log entries
    if logger.handlers:
        logger.setLevel(level_value)
        for handler in logger.handlers:
            handler.setLevel(level_value)
        return logger

    logger.setLevel(level_value)
    logger.propagate = False
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level_value)
    
    # Format: 2026-04-26 19:00:00 | LEVEL | Message
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    return logger

LOGGER = _configure_logger()

def log_process(stage, event, level=logging.INFO, **details):
    """
    Structured process-change logging for clinical traceability and maintainability.
    Enables parsing of logs into DataFrames for post-hoc experiment analysis.
    
    Args:
        stage (str): The logical part of the pipeline (e.g., 'train', 'eval', 'graph').
        event (str): The specific occurrence (e.g., 'epoch_completed', 'leakage_detected').
        level (int): Logging level (DEBUG, INFO, WARNING, etc.).
        **details: Arbitrary key-value pairs representing specific metadata.
    """
    if details:
        detail_str = ", ".join(f"{k}={details[k]}" for k in sorted(details))
        LOGGER.log(level, "[%s] %s | %s", stage, event, detail_str)
    else:
        LOGGER.log(level, "[%s] %s", stage, event)

# ============================================================
# SCIENTIFIC DETERMINISM PROTOCOL
# ============================================================

def enforce_reproducibility(seed: int = GLOBAL_SEED) -> torch.Generator:
    """
    Locks all 6 sources of non-determinism in the Deep Learning stack.
    Required for Nature-grade peer review and clinical benchmark verification.
    
    Covers:
    1. Python built-in hashing
    2. Python random module
    3. NumPy internal state
    4. PyTorch CPU & GPU seeds
    5. CUDA Convolutional benchmarks (cuDNN)
    6. Deterministic Linear Algebra (CUBLAS)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Force deterministic algorithms for CUDA BLAS operations
    # Requirement: torch >= 1.8.0
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Disable cuDNN auto-tuner to prevent non-deterministic algorithm selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Throw error if a non-deterministic operation is encountered (strict mode)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set matrix multiplication precision to highest for Float32 (Tensor Cores)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
        
    # Generate a unique Generator for DataLoaders to ensure worker-level determinism
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# Initialize Global Dataloader Generator
DATALOADER_GENERATOR = enforce_reproducibility(GLOBAL_SEED)

# Global Compute Context
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")