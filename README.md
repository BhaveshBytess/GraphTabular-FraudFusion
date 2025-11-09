# Graph-Tabular Fusion Extension

**Fusion-only extension** that reuses baseline splits and metrics to evaluate `[tabular features || graph embeddings] → XGBoost`.

## Overview

This repository extends the baseline fraud detection work by implementing a **graph-tabular fusion model** that combines:
- **Graph embeddings** (Node2Vec or GraphSAGE exports)
- **Tabular features** (Local AF1-93 or Full AF1-182)
- **XGBoost classifier** as the fusion learner

**Key principles:**
- ✅ Reuses baseline temporal splits (no leakage)
- ✅ Imports baseline metrics for comparison
- ✅ Does NOT retrain baseline GNN/tabular models
- ✅ Follows PROJECT_SPEC v2 strictly

## Repository Structure

```
graph-tabular-fusion/
├── data/
│   └── Elliptic++ Dataset/         # Dataset files (user-provided)
├── notebooks/
│   ├── 01_generate_embeddings.ipynb
│   ├── 02_fusion_xgb.ipynb
│   └── 03_ablation_studies.ipynb
├── src/
│   ├── data/                       # Data loaders and utilities
│   ├── embeddings/                 # Node2Vec, GraphSAGE export
│   ├── train/                      # XGBoost fusion trainer
│   ├── utils/                      # Metrics, seeding, logging
│   └── eval/                       # Comparison reports
├── configs/                        # YAML configurations
├── reports/                        # Metrics and plots
├── docs/                           # Specs and provenance
└── tools/                          # Import utilities
```

## Quick Start

### 1. Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Elliptic++ dataset files

### 2. Installation

```bash
pip install -r requirements.txt
```

### 3. Verify Dataset

```bash
python src/data/verify_dataset.py "data/Elliptic++ Dataset"
```

### 4. Generate Embeddings

```bash
# Option A: Node2Vec (fast, unsupervised)
jupyter notebook notebooks/01_generate_embeddings.ipynb
```

### 5. Train Fusion Model

```bash
# XGBoost on fused features
jupyter notebook notebooks/02_fusion_xgb.ipynb
```

## Baseline Provenance

This extension imports artifacts from:
- **Repository:** https://github.com/BhaveshBytess/Revisiting-GNNs-FraudDetection
- **Commit:** ccab3f9ff99c1c84090a396015ed695fa8394c39
- **Imported:** Temporal split logic, baseline metrics CSV, utility modules

See `docs/baseline_provenance.json` for full details.

## Fusion Protocols

### Protocol A (Default)
- **Features:** Local only (AF1-93) + Graph embeddings
- **Rationale:** Avoids double-encoding neighbor information

### Protocol B (Comparison)
- **Features:** All features (AF1-182) + Graph embeddings
- **Rationale:** Full feature set, may have redundancy

## Results

See `reports/metrics_summary.csv` for consolidated baseline vs fusion comparison.

## Citation

If you use this work, please cite:
- Elliptic++ dataset
- Baseline repository (see provenance)

## License

Educational/demonstrative use. Respect Elliptic++ dataset terms.

---

**Status:** ✅ Initialized per PROJECT_SPEC v2
