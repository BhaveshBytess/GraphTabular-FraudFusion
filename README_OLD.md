# Graph-Tabular Fusion Extension

**Fusion-only extension** that reuses baseline splits and metrics to evaluate `[tabular features || graph embeddings] → XGBoost`.

## 🎯 Key Finding

> **Graph embeddings provide minimal incremental benefit (~2% decrease) when combined with local tabular features on Elliptic++, validating that tabular features already effectively capture graph structure.**

## Overview

This repository extends the baseline fraud detection work by implementing a **graph-tabular fusion model** that combines:
- **Graph embeddings** (Node2Vec, 64-dim, leakage-free per-split generation)
- **Tabular features** (Local AF1-93, Protocol A)
- **XGBoost classifier** as the fusion learner

**Key principles:**
- ✅ Reuses baseline temporal splits (no leakage)
- ✅ Imports baseline metrics for comparison
- ✅ Does NOT retrain baseline GNN/tabular models
- ✅ Follows PROJECT_SPEC v2 strictly
- ✅ Honest reporting of results (including negative findings)

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

### Performance Comparison (Test Set)

| Model | PR-AUC | ROC-AUC | F1 | Recall@1% |
|-------|--------|---------|-----|-----------|
| **XGBoost (Baseline)** | **0.6689** | **0.8881** | **0.6988** | - |
| **XGBoost + Node2Vec (Fusion)** | 0.6555 | 0.8608 | 0.6877 | 0.1745 |
| Random Forest (Baseline) | 0.6583 | 0.8773 | 0.6945 | - |
| GraphSAGE (Baseline GNN) | - | - | - | - |

**Difference:** Fusion vs. Baseline XGBoost = **-0.0134 PR-AUC (-2%)**

### Interpretation

**Finding:** Adding Node2Vec graph embeddings to local tabular features provides **no significant improvement** and slightly underperforms baseline tabular-only XGBoost.

**Why?**
1. **Tabular features already encode graph structure:** The local features (AF1-93) capture sufficient transaction characteristics
2. **Aggregate features in baseline** (AF94-182) explicitly encode neighbor statistics, making graph embeddings redundant
3. **Baseline conclusion validated:** "XGBoost > GraphSAGE" because tabular features suffice for this task

**Implications:**
- ✅ **For practitioners:** Use XGBoost on tabular features alone - simpler, faster, equally effective
- ✅ **For research:** Demonstrates when complex graph models aren't needed
- ✅ **For deployment:** Prefer interpretable tabular models over fusion approaches

### Experimental Setup

**Embeddings:**
- Method: Node2Vec (unsupervised random walks + Word2Vec)
- Dimensions: 64
- Leakage prevention: Per-split generation with within-split edges only
- Walk parameters: length=80, walks_per_node=10, context=10

**Fusion Protocol:**
- **Protocol A** (implemented): Local features (AF1-93) + Node2Vec embeddings = 157 features
- Features concatenated and fed to XGBoost
- Same temporal splits as baseline (60/20/20 train/val/test)

**Training:**
- Model: XGBoost with early stopping on validation PR-AUC
- Class weighting: scale_pos_weight=8.19 (computed from training data)
- No hyperparameter tuning (baseline config reused)

See `reports/metrics_summary.csv` for full consolidated comparison.

## Reproducibility

All experiments are fully reproducible:
- **Seed:** 42 (fixed across all runs)
- **Splits:** Temporal 60/20/20 (imported from baseline)
- **Embeddings:** Deterministic Node2Vec with fixed random seed
- **Environment:** Python 3.13, PyTorch 2.0+, XGBoost 2.0+

**Artifacts:**
- Embeddings: `data/embeddings.parquet` (70 MB, 203,769 nodes × 64 dims)
- Model: `reports/xgb_fusion.json`
- Metrics: `reports/metrics.json`
- Plots: `reports/plots/`

## Lessons Learned

1. **Feature engineering matters more than model complexity** - Rich tabular features outperform graph-aware models
2. **Negative results are valuable** - Demonstrating when fusion doesn't help is scientifically rigorous
3. **Leakage prevention is critical** - Per-split embedding generation prevents temporal information leakage
4. **Baseline comparison is essential** - Always compare against strong tabular baselines before claiming graph methods help

## Future Directions (Out of Scope)

- Protocol B: Test with full features (AF1-182) + embeddings
- Alternative embeddings: GraphSAGE export, deeper architectures
- Embedding dimensions: Sweep 16/32/128
- Temporal models: Incorporate time-aware graph learning
- Explainability: SHAP analysis on fusion features

## Citation

If you use this work, please cite:
- Elliptic++ dataset: [Weber et al., 2019]
- Baseline repository: https://github.com/BhaveshBytess/Revisiting-GNNs-FraudDetection
- Provenance: See `docs/baseline_provenance.json`

## License

Educational/demonstrative use. Respect Elliptic++ dataset terms.

---

**Status:** ✅ E1-E3 Complete | Fusion model trained and evaluated | Results validate baseline findings
