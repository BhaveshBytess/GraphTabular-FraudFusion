# Release Notes - v1.0.0

**Release Date:** November 9, 2025  
**Project:** Graph-Tabular Fusion on Elliptic++ Bitcoin Fraud Detection  
**Repository:** https://github.com/BhaveshBytess/GraphTabular-FraudFusion

---

## 🎯 Key Finding

**Graph embeddings provide minimal benefit when tabular features already encode graph structure.**

- **Baseline XGBoost (tabular-only):** PR-AUC 0.6689 🏆
- **Fusion (XGBoost + Node2Vec):** PR-AUC 0.6555 (-2%)
- **Conclusion:** Rich tabular features > explicit graph embeddings

This negative result is **scientifically valuable** and guides practitioners away from unnecessary architectural complexity.

---

## ✨ What's Included

### 🔬 Core Functionality
- ✅ **Node2Vec embedding generation** (leakage-free, per-split)
- ✅ **XGBoost fusion trainer** with early stopping
- ✅ **Comprehensive evaluation** (PR-AUC, ROC-AUC, F1, Recall@K)
- ✅ **Temporal split logic** (60/20/20 from baseline)
- ✅ **Dataset verification** utilities
- ✅ **Baseline comparison** framework

### 📊 Visualizations (7 publication-quality plots)
1. **model_comparison.png** - Multi-metric bar charts
2. **fusion_vs_baseline.png** - Direct comparison
3. **pipeline_diagram.png** - Architecture overview
4. **feature_contribution.png** - Feature impact
5. **summary_table.png** - Results summary
6. **pr_auc_comparison.png** - PR-AUC comparison
7. **f1_comparison.png** - F1 comparison

### 📚 Documentation
- ✅ **Professional README** with badges, figures, and detailed analysis
- ✅ **FINAL_SUMMARY.md** - Comprehensive project report
- ✅ **CHANGELOG.md** - Version history
- ✅ **CITATION.cff** - Citation guidelines
- ✅ **LICENSE** - MIT License
- ✅ **Baseline provenance** tracking

### 📓 Notebooks (Kaggle-ready)
1. `01_generate_embeddings.ipynb` - Embedding generation pipeline
2. `02_fusion_xgb.ipynb` - Fusion model training
3. `03_ablation_studies.ipynb` - Optional experiments (placeholder)

### ⚙️ Configuration
- `embed_node2vec.yaml` - Node2Vec parameters
- `embed_graphsage.yaml` - GraphSAGE export (stub)
- `fusion_xgb.yaml` - Fusion model configuration

---

## 📈 Results Summary

### Test Set Performance

| Metric | Baseline (XGBoost) | Fusion (XGBoost+Node2Vec) | Difference |
|--------|-------------------|---------------------------|------------|
| **PR-AUC** | **0.6689** 🏆 | 0.6555 | -0.0134 (-2%) |
| **ROC-AUC** | **0.8881** | 0.8608 | -0.0273 (-3%) |
| **F1** | **0.6988** | 0.6877 | -0.0111 (-2%) |
| **Recall@1%** | - | 0.1745 | - |

### Key Insights

1. **Tabular features already encode graph structure**
   - Local features (AF1-93) capture transaction characteristics
   - Baseline aggregates (AF94-182) explicitly encode neighbor info

2. **Node2Vec embeddings are redundant**
   - Random walk-based embeddings learn similar patterns
   - No unique signal beyond tabular representation

3. **Simpler is better**
   - XGBoost on features alone preferred
   - Faster, more interpretable, equally effective

---

## 🔬 Experimental Rigor

### Reproducibility ✅
- **Seed:** 42 (fixed for all operations)
- **Splits:** Temporal 60/20/20 (imported from baseline)
- **Deterministic:** All random operations controlled
- **Documentation:** Complete step-by-step reproduction guide

### Leakage Prevention ✅
- **Per-split embeddings:** Train/val/test computed independently
- **Within-split edges only:** No cross-split information
- **Temporal isolation:** No future information leakage

### Fair Comparison ✅
- **Same splits** as baseline (exact txId alignment)
- **Same metrics** (PR-AUC, ROC-AUC, F1, Recall@K)
- **Same class weighting** (computed from training data)
- **No hyperparameter tuning** (baseline config reused)

---

## 💻 Technical Specifications

### Environment
- **Python:** 3.8+
- **PyTorch:** 2.0+
- **XGBoost:** 2.0+
- **NetworkX:** 3.0+
- **Gensim:** 4.3+

### Dataset
- **Name:** Elliptic++
- **Nodes:** 203,769 Bitcoin transactions
- **Edges:** 234,355 transaction flows
- **Features:** 166 per node (local + aggregates)
- **Labels:** Fraud (4,545) / Legit (42,019) / Unknown (157,205)

### Model Architecture
- **Embeddings:** Node2Vec (64-dim, 80-length walks, 10 per node)
- **Features:** 93 local + 64 embeddings = 157 total
- **Learner:** XGBoost with early stopping
- **Training time:** ~2 minutes (after embeddings)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/BhaveshBytess/GraphTabular-FraudFusion.git
cd GraphTabular-FraudFusion

# Setup environment
pip install -r requirements.txt

# Verify dataset (download from Google Drive)
python src/data/verify_dataset.py "data/Elliptic++ Dataset"

# Generate embeddings (~30 min CPU)
python scripts/generate_embeddings.py

# Train fusion model (~2 min)
python scripts/train_fusion.py

# View results
ls reports/  # Metrics, model, plots
```

---

## 📁 Repository Structure

```
GraphTabular-FraudFusion/
├── data/                    # Dataset + embeddings
├── notebooks/               # Kaggle-ready notebooks (3)
├── src/                     # Source code
│   ├── data/               # Loaders, verification
│   ├── embeddings/         # Node2Vec implementation
│   ├── train/              # XGBoost fusion
│   ├── eval/               # Comparison reports
│   └── utils/              # Utilities
├── configs/                # YAML configurations (3)
├── reports/                # Results, metrics, plots
├── scripts/                # Execution pipelines
├── docs/                   # Specifications, provenance
└── README.md               # Main documentation
```

---

## 🎓 Scientific Contribution

This work demonstrates:

1. **When graph methods don't help** - Empirical evidence
2. **Importance of strong baselines** - Always compare with tabular methods
3. **Value of negative results** - Scientifically rigorous reporting
4. **Reproducible methodology** - Complete pipeline and documentation
5. **Practical guidance** - Actionable insights for practitioners

**Publication-worthy aspects:**
- Leakage-free temporal evaluation
- Comprehensive baseline comparison
- Clear negative result interpretation
- Reproducible experimental design

---

## 🙏 Acknowledgments

- **Elliptic** for the Elliptic++ dataset
- **Baseline project** for splits, metrics, and foundational work
- **PyTorch Geometric**, **XGBoost**, **NetworkX**, **Gensim** communities
- Open-source ML community

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

Dataset subject to Elliptic++ terms and conditions.

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/BhaveshBytess/GraphTabular-FraudFusion/issues)
- **Repository:** https://github.com/BhaveshBytess/GraphTabular-FraudFusion

---

## 🔖 Version Tags

- **v1.0.0** - Initial public release (November 9, 2025)

---

**Status:** ✅ Complete | 📊 Results validated | 🎓 Portfolio-ready | ⭐ Zenodo-eligible

---

## 📦 Files Included in Release

**Source Code:**
- Complete Python source code (22 modules)
- Notebooks (3 Kaggle-ready)
- Configuration files (3 YAML)
- Scripts (4 execution pipelines)

**Documentation:**
- README.md (comprehensive)
- FINAL_SUMMARY.md
- CHANGELOG.md
- CITATION.cff
- LICENSE
- INITIALIZATION_REPORT.md

**Results:**
- Metrics (JSON, CSV)
- Trained model (XGBoost)
- Visualizations (7 PNG plots, 300 DPI)

**Metadata:**
- Baseline provenance tracking
- Git history (4 commits)
- Requirements specification

---

**Total Archive Size:** ~1 MB (excluding dataset and embeddings)

**Zenodo DOI:** *To be assigned upon release publication*

---

**Thank you for your interest in this project!** ⭐

If you find this work useful, please consider:
- ⭐ **Starring the repository**
- 📖 **Citing in your work**
- 🔗 **Sharing with others**
- 💬 **Opening issues for questions**

---

**End of Release Notes**
