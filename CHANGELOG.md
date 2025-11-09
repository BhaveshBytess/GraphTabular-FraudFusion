# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-09

### Added
- Initial release of Graph-Tabular Fusion project
- Node2Vec embedding generation with leakage-free per-split computation
- XGBoost fusion model trainer
- Comprehensive evaluation metrics (PR-AUC, ROC-AUC, F1, Recall@K)
- 7 publication-quality visualizations (300 DPI)
- Complete documentation and README
- Temporal split logic from baseline (60/20/20)
- Dataset verification utilities
- Baseline provenance tracking
- Kaggle-ready notebooks (3)
- Configuration files (YAML)
- FINAL_SUMMARY.md with complete analysis

### Key Findings
- XGBoost baseline: PR-AUC 0.6689
- Fusion (XGBoost + Node2Vec): PR-AUC 0.6555 (-2%)
- Graph embeddings provide minimal benefit when tabular features already encode structure
- Validates baseline finding: rich features > architectural complexity

### Features
- Reproducible experiments (seed 42)
- Leakage prevention (per-split embeddings)
- Fair baseline comparison
- Honest negative result reporting
- Publication-quality visualizations
- Professional documentation

### Technical Details
- Python 3.8+
- PyTorch 2.0+
- XGBoost 2.0+
- Node2Vec (NetworkX + Gensim)
- 203,769 nodes, 234,355 edges
- 64-dimensional embeddings
- 157 total features (93 tabular + 64 embeddings)

## [Unreleased]

### Potential Future Work
- Protocol B implementation (AF1-182 + embeddings)
- Embedding dimension sweep (16/32/128)
- GraphSAGE export comparison
- MLP fusion learner
- SHAP explainability analysis
- Temporal embedding models
- Cross-dataset evaluation

---

## Version History

- **v1.0.0** (2025-11-09): Initial public release with complete fusion pipeline and results
