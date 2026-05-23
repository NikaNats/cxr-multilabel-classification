from __future__ import annotations

import datetime
import hashlib
import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ==============================================================================
# § 1  EXPERIMENT PROVENANCE & IDENTIFICATION
# ==============================================================================

GLOBAL_SEED: int = 42
EXPERIMENT_NAME: str = "CXR_Synapse_Foundation_ChestMNIST"

# Path configuration
OUTPUT_DIR: Path = Path("runs") if hasattr(sys, "frozen") else Path("./runs")
LOG_DIR: Path = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Unique Run Fingerprint: Generates an immutable, traceable 12-char SHA-256 hash.
# Combines name, seed, hardware specs, and timestamp to prevent run collisions.
_PROVENANCE_STR: str = (
    f"{EXPERIMENT_NAME}_{GLOBAL_SEED}_"
    f"{platform.processor()}_"
    f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}"
)
EXPERIMENT_ID: str = hashlib.sha256(_PROVENANCE_STR.encode()).hexdigest()[:12]

# ==============================================================================
# § 2  OBSERVABILITY, TELEMETRY & RICH LOGGING FORMATTERS
# ==============================================================================

def _configure_structured_logger(
    name: str = "cxr_synapse", level: int | str | None = None
) -> logging.Logger:
    """
    Creates a process-wide logger with a dual-handler setup (Stream + File).
    All logs are timestamped with ISO-8601 formatting for clinical audatability.
    """
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

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_value)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file_path = LOG_DIR / f"run_{EXPERIMENT_ID}.log"
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level_value)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


LOGGER: logging.Logger = _configure_structured_logger()


def log_process(stage: str, event: str, level: int = logging.INFO, **details: Any) -> None:
    """Standard logging with key=value details."""
    if details:
        detail_str = ", ".join(f"{k}={details[k]}" for k in sorted(details))
        LOGGER.log(level, "[%s] %s | %s", stage.upper(), event, detail_str)
    else:
        LOGGER.log(level, "[%s] %s", stage.upper(), event)


def log_clinical_report(
    stage: str, report_title: str, content: str, level: int = logging.INFO
) -> None:
    """
    Prints a highly structured, clinical-grade multi-line report block into the logs.
    Ensures supreme readability for post-hoc forensic audits.
    """
    border = "═" * 78
    header = f"║  {report_title.upper()} — RUN: {EXPERIMENT_ID}"
    padding = " " * (78 - len(header) - 1)
    
    formatted_report = (
        f"\n{border}\n"
        f"{header}{padding}║\n"
        f"{'╟' + '─' * 76 + '╢'}\n"
    )
    for line in content.strip().split("\n"):
        formatted_report += f"║ {line:<75} ║\n"
    formatted_report += f"{border}\n"
    
    LOGGER.log(
        level, "[%s] Clinical Report Generated:\n%s", stage.upper(), formatted_report
    )


# ==============================================================================
# § 2.1  ASCII FORENSIC CHART GENERATORS (Terminal-Based Visualization)
# ==============================================================================

def format_ascii_matrix(
    matrix: np.ndarray, labels: list[str], title: str = "MATRIX"
) -> str:
    """
    Translates a 2D matrix (e.g., Correlation, Cramer's V) into a perfectly aligned
    ASCII representation for logging.
    """
    n = matrix.shape[0]
    short_labels = [lbl[:8] for lbl in labels]
    
    # Header row
    col_headers = " " * 10 + " | ".join(f"{lbl:>8}" for lbl in short_labels)
    output = [
        f"=== ASCII HEATMAP: {title.upper()} ===",
        col_headers,
        "-" * len(col_headers),
    ]
    
    for i in range(n):
        row_str = f"{short_labels[i]:<8} | "
        cells = []
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                cells.append(f"{'NaN':>8}")
            else:
                cells.append(f"{val:>8.3f}")
        row_str += " | ".join(cells)
        output.append(row_str)
    output.append("-" * len(col_headers))
    return "\n".join(output)


def format_ascii_histogram(
    data: np.ndarray, bins: int = 10, width: int = 30, title: str = "DISTRIBUTION"
) -> str:
    """
    Generates a text-based ASCII histogram of continuous data (e.g., SNR, Entropies)
    allowing immediate visual assessment of skewness directly inside the log file.
    """
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return f"=== ASCII HISTOGRAM: {title.upper()} ===\nNo valid data to plot."
        
    counts, edges = np.histogram(valid_data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    
    output = [f"=== ASCII HISTOGRAM: {title.upper()} ==="]
    for i in range(bins):
        bar = "█" * int((counts[i] / max_count) * width)
        # Pad shorter bars with thin line for visual guidance
        if len(bar) < width:
            bar += "░" * (width - len(bar))
        bin_label = f"[{edges[i]:6.2f} : {edges[i + 1]:6.2f}]"
        output.append(f"{bin_label} | {bar} | n={counts[i]:<5}")
    output.append(f"Total samples plotted: {len(valid_data):,}")
    return "\n".join(output)


# ==============================================================================
# § 3  HARDWARE DIAGNOSTICS & TELEMETRY
# ==============================================================================

def get_hardware_telemetry() -> dict[str, Any]:
    """
    Queries the active compute environment to catalog hardware specs.
    Enforces TensorFloat-32 (TF32) precision on Ampere+ architectures.
    """
    telemetry = {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if telemetry["cuda_available"]:
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        total_memory = (
            torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        )
        
        telemetry.update(
            {
                "device_name": device_name,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "total_vram_gb": f"{total_memory:.2f}",
            }
        )

        if capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            telemetry["tf32_enabled"] = True
        else:
            telemetry["tf32_enabled"] = False
    else:
        telemetry["device_name"] = "CPU"
        
    return telemetry


# ==============================================================================
# § 4  SCIENTIFIC DETERMINISM PROTOCOL
# ==============================================================================

def enforce_reproducibility(seed: int = GLOBAL_SEED) -> torch.Generator:
    """
    Locks all 6 sources of non-determinism in the Deep Learning stack.
    Required for Nature-grade peer review and clinical benchmark verification.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ==============================================================================
# § 5  INITIALIZATION & SYSTEM DIAGNOSTIC REPORT
# ==============================================================================

DATALOADER_GENERATOR = enforce_reproducibility(GLOBAL_SEED)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HW_TELEMETRY = get_hardware_telemetry()
log_process(
    "config",
    "experiment_initialized",
    run_id=EXPERIMENT_ID,
    seed=GLOBAL_SEED,
    device=_HW_TELEMETRY["device_name"],
    vram_gb=_HW_TELEMETRY.get("total_vram_gb", "N/A"),
    tf32=_HW_TELEMETRY.get("tf32_enabled", False),
    log_file=f"runs/logs/run_{EXPERIMENT_ID}.log",
)