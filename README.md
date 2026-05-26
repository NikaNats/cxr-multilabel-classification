# CXR-Synapse: Graph-Guided Pathology Attention Foundation Model

This repository contains the official implementation of **CXR-Synapse**, a clinical-grade foundation framework for multi-label chest X-ray (CXR) classification. The architecture routes localized chest representations using pathology-specific RadLex queries, 2D Rotary Position Embeddings (2D RoPE), and structured clinical graph adjacency maps. 

The pipeline includes a hardware-accelerated forensic dataset auditor to eliminate cross-split data leakage, evaluate probability calibration via Asymmetric Isotonic Regression (AIR), and assess risk control boundaries using conformal prediction.

## Technical Overview

The execution pipeline consists of a sequential, frozen-backbone paradigm:
1. **Feature Extraction (`Extraction/`)**: Isolated extraction of 1376-dimensional spatial features using a dedicated TensorFlow container.
2. **Forensic Audit (`audit.py`)**: Parallelized bitwise hashing (Blake2b) to systematically isolate and remove cross-split duplicates, followed by GPU-accelerated spatial morphology profiling.
3. **Cardinality Analysis (`analyze_cardinality.py`)**: Evaluates multi-label pathology distribution across the split boundaries.
4. **Model Training (`train.py`)**: Optimizes the CXR-Synapse Foundation model under class-balanced asymmetric loss (CB-ASL) and SWA optimization.
5. **Diagnostic Suite (`visualizer.py`)**: Generates scientific-grade diagnostic plots (Decision Curve Analysis, UMAP manifolds, calibration curves, and attention maps).


## Environment Setup

Due to differing dependency requirements for tensorflow-text compilation and the PyTorch SWA training suite, the pipeline is divided into two container environments.

### Environment A: Base Clinical Model Container (NVIDIA NGC)
Used for forensic auditing, model training, and diagnostic visualization. Run the container from your host terminal, mapping your project directory to `/workspace`:

```bash
docker run --gpus all -it --rm \
  -v "/path/to/your/cxr_project:/workspace" \
  -p 8888:8888 \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:26.03-py3
```

Once inside the container shell, install the processing and visualization dependencies:
```bash
# Core medical, statistical, and plotting libraries
pip install medmnist torchcp transformers sentencepiece scikit-image seaborn pandas timm opencv-python-headless umap-learn statsmodels pypng

# CUDA-accelerated auditing backends
pip install cupy-cuda12x cucim-cu12
```

### Environment B: Dedicated Feature Extraction Container (TensorFlow GPU)
Used exclusively for running the heavy feature extraction pipeline. Launch this container separately:

```bash
docker run --gpus all -it --rm \
  -v "/path/to/your/cxr_project:/tf/notebooks" \
  tensorflow/tensorflow:2.20.0-gpu-jupyter \
  bash -c 'apt-get update && apt-get install -y libcudnn9-cuda-12 && ldconfig && apt-get remove -y libcudnn8 && pip install medmnist transformers tensorflow-text==2.20.0 opencv-python-headless torch tqdm && exec bash'
```

## Execution Pipeline

Follow these steps in sequential order to run the entire project from raw dataset to final diagnostic outputs.

### Step 1: Feature Extraction (Environment B)
Inside the interactive shell of your **TensorFlow GPU Container**, navigate to the extraction directory and run the pipeline to generate your spatial feature representations:

```bash
cd /tf/notebooks/Extraction
python extraction_pipeline.py
```
*Outputs: Generates `train_embeddings.pt`, `val_embeddings.pt`, and `test_embeddings.pt` containing 1376-dimensional embeddings inside `/workspace/Extraction/cxr_embeddings_10percent`.*

---

*All subsequent steps are performed inside the **NVIDIA NGC Container** (Environment A) at `/workspace`:*

```bash
cd /workspace
```

### Step 2: Pre-run Forensic Verification & Audit
Run the database deduplication audit to sweep and clean validation and test splits of any historical train overlap, while generating spatial-variance and comorbidity profiles:
```bash
python audit.py
```
*Outputs: Generates `forensic_audit_train.pdf`, `forensic_audit_val.pdf`, and `forensic_audit_test.pdf` containing empirical comorbidity heatmaps and edge-density profiles.*

### Step 3: Analyze Splitting Cardinalities
Evaluate the joint probability distributions of co-occurring pathologies across your newly cleaned dataset splits:
```bash
python analyze_cardinality.py
```

### Step 4: Train the CXR-Synapse Foundation Network
Begin training the main model using graph-guided cross-attention, class-balanced loss weights (derived via Effective Number of Samples), and Stochastic Weight Averaging (SWA):
```bash
python train.py
```
*Outputs: Evaluates validation macro-AUROC at each epoch and saves the SWA state dictionary to `CXR_Synapse_Foundation_Seed_[seed].pth`.*

### Step 5: Generate Diagnostic & Clinical Suite
Execute the main visualization and conformal evaluation script to output publication-ready figures mapping latent spaces and spatial attention maps:
```bash
python visualizer.py
```
*Outputs: High-resolution PDF/PNG panels depicting Macro-ROC, Isotonic Calibration curves, Conformal coverage trade-offs, and localized PaQ attention overlays.*

## Citation & Academic Reference

If you utilize this framework, the associated dataset deduplication protocols, or the CXR-Synapse architecture in your research, please cite the work using the academic reference below:

```bibtex
@mastersthesis{cxrsynapse2025,
  author    = {Author Name},
  title     = {CXR-Synapse: Clinical Graph-Guided Multi-Label Attention Foundations for Thoracic Imaging},
  school    = {Master's Thesis Project},
  year      = {2025}
}
```