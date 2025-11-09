Perfect—here’s your **trimmed, extension-only** spec for **Direction B** that *reuses* your baseline repo outputs and focuses **only** on the fusion model. It keeps your structure, but removes re-training of old baselines/GNNs and adds clear pointers to import prior artifacts.

You can paste this as `docs/PROJECT_SPEC.md` in the new repo (`graph-tabular-fusion`).

---

# PROJECT_SPEC (v2 — Graph–Tabular Fusion, **Extension of Baseline**)

## 0) Purpose (single source of truth)

Define the **what** of this extension: minimal scope to deliver a **fusion model** that combines graph embeddings with tabular features on Elliptic++. This project **reuses baseline artifacts** (splits + baseline metrics) from the previously completed repo and **does not** re-train baseline GNN/tabular models. All tasks align with this spec.

---

## 1) Goal & Scope

**Project:** `graph-tabular-fusion`
**Goal:** Implement a clean, reproducible **fusion baseline** by concatenating **graph embeddings** (GraphSAGE/Node2Vec/GCN export) with **tabular features** and training **XGBoost/MLP**—using the **same temporal splits** and label policy as the baseline repo.
**Audience:** Recruiters, collaborators, future-you.
**Deliverable type:** Portfolio/demo repo — readable notebooks first, reusable `src/` utilities second.

**In scope**

* Import **splits.json** and **metrics_summary.csv** from the baseline repo (source of truth).
* Generate per-node **graph embeddings** (choose fastest valid path):

  * **Option A (fast, no labels):** Node2Vec.
  * **Option B (reuse):** Export embeddings from **existing GraphSAGE checkpoint** in baseline repo (penultimate layer).
  * **Option C (light):** Train a small GraphSAGE just to export embeddings (if checkpoint missing).
* Build **fusion datasets**:

  * Protocol A (**Local-only** AF1–AF93 + embeddings) — default, avoids double-encoding.
  * Protocol B (Local+Aggregates AF1–AF182 + embeddings) — for comparison.
* Train **tabular learners** on fused features: **XGBoost (primary)**, **MLP (optional)**.
* Evaluate with the **same metrics** and **temporal splits**; produce a **side-by-side table** that includes imported baseline metrics vs fusion results.

**Out of scope (for this repo)**

* Re-training baseline GNNs (GCN/GraphSAGE/GAT) or tabular baselines (already done).
* Temporal memory models, hetero/hypergraph, curvature/geometry, advanced explainers.
* Any synthetic/mock data.
* Productionization (APIs/serving).

---

## 2) Dataset (Elliptic++)

**Identity:** Elliptic++ Bitcoin transaction graph (nodes = transactions; edges = directed flows).
**Location:** `data/Elliptic++ Dataset/` (local only; user provides files).
**Download:** [https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l)

**Required files**

* `txs_features.csv` — `txid`, `timestamp`, **Local_feature_1..93**, **Aggregate_feature_1..89** (total 182)
* `txs_classes.csv` — `txid`, `class` (1=fraud, 2=legit, 3=unknown)
* `txs_edgelist.csv` — `txId1`, `txId2`

**Data policy**

* **No synthetic data.** Stop if files/columns mismatch; request correct path.
* Notebooks must verify file presence before running.
* Any preprocessing is deterministic and logged.

---

## 3) Temporal Split (no leakage)

**Source of truth:** Reuse **baseline** `splits.json`.

* Train/Val/Test membership **identical** to baseline.
* For **embedding generation**, enforce **split isolation**: for each split, include only edges whose endpoints both lie in that split (avoid future leakage).
* Persist counts and boundaries in this repo (copy of baseline `splits.json` under `data/Elliptic++ Dataset/`).

---

## 4) Preprocessing & Features

* Map `txid` → contiguous indices `[0..N-1]` (persist mapping; must match splits).
* Filter edges to known nodes; coalesce duplicates.
* Optional normalization (fit scalers on **train only**; apply to val/test).
* **Fusion protocols:**

  * **Protocol A (default):** **Local (AF1–AF93)** + **Embeddings** (safer, avoids double-encoding).
  * **Protocol B:** **All features (AF1–AF182)** + **Embeddings** (document redundancy, if any).

---

## 5) Models

### 5.1 Graph encoders (embedding generators)

Choose **one** primary path for MVP; keep others as fallbacks:

* **Node2Vec** (fast, unsupervised, good baseline).
* **GraphSAGE export** from existing baseline checkpoint (penultimate hidden states).
* **GraphSAGE (light train)** if export is unavailable.

**Output:** per-node embedding matrix `H ∈ R^{N×d}` saved (`.parquet`/`.npy`) with `txid` alignment.

### 5.2 Fusion learners

* **XGBoost (primary)**
* **MLP (optional)**

**Inputs:**

* **Fusion:** `[X_tab || H]`
* (Optional comparisons) **Tabular-only** (`X_tab`) and **Embeddings-only** (`H`) can be run cheaply if needed, but **not required** if already known from baseline analysis.

**Output:** probability of class 1 (fraud).

---

## 6) Training & Evaluation

**Loss**

* Binary cross-entropy (or multi-class CE with masking); be consistent.
* Use `pos_weight` or class weights computed from **train** labels.

**Optimization**

* XGBoost: early stopping on **val** AUC-PR; typical ranges in config.
* MLP: Adam (`lr`≈1e-3), `weight_decay`≈5e-4, early stopping on **val PR-AUC**.

**Evaluation protocol**

* Threshold selected on **val** to maximize F1; reuse on **test**.
* Report on **test**:

  * PR-AUC (primary), ROC-AUC, F1, Recall@K where K ∈ {0.5%, 1%, 2%}.
* **Side-by-side comparison**:

  * Import **baseline** metrics (tabular-only + GNNs) from previous repo’s `reports/metrics_summary.csv`.
  * Append **fusion** rows computed here.
  * Produce a consolidated comparison table + PR/ROC plots.

**Artifacts**

* `checkpoints/fusion_best.pt` (if MLP used)
* `reports/metrics.json`
* `reports/plots/*.png` (PR/ROC + ablation bars)
* Append to `reports/metrics_summary.csv`:

  * `timestamp, experiment, model, split, pr_auc, roc_auc, f1, recall@1%`

---

## 7) Reproducibility

Always:

```python
from src.utils.seed import set_all_seeds
set_all_seeds(seed)

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

* Save `splits.json`, scaler params, library versions.
* Avoid absolute paths; use project-relative paths.
* **Provenance lock:** `docs/baseline_provenance.json` with:

  * baseline repo URL, commit SHA, Zenodo DOI, date imported.

---

## 8) Repository Scaffold

```
graph-tabular-fusion/
│
├── data/
│   └── Elliptic++ Dataset/            # user-provided dataset + baseline splits.json
│
├── notebooks/
│   ├── 01_generate_embeddings.ipynb   # Node2Vec or GraphSAGE-export
│   ├── 02_fusion_xgb.ipynb            # [X_tab || H] → XGB
│   └── 03_ablation_studies.ipynb      # Protocol A vs B; (optional) d-sweeps
│
├── src/
│   ├── data/
│   │   ├── elliptic_loader.py         # masks + splits (reused)
│   │   ├── verify_dataset.py          # --check CLI
│   │   └── merge_embeddings.py        # left-join H with features on txid
│   ├── embeddings/
│   │   ├── node2vec.py                # fast MVP
│   │   └── graphsage_export.py        # export from baseline checkpoint
│   ├── train/
│   │   └── fusion_xgb.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── seed.py
│   │   └── logger.py
│   └── eval/
│       └── fusion_report.py           # side-by-side table + plots
│
├── configs/
│   ├── embed_node2vec.yaml
│   ├── embed_graphsage.yaml
│   └── fusion_xgb.yaml
│
├── scripts/
│   ├── 10_embed_node2vec.sh
│   ├── 20_export_graphsage.sh
│   └── 30_train_fusion.sh
│
├── reports/
│   ├── plots/
│   └── metrics_summary.csv
│
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── AGENT.md
│   ├── START-PROMPT.md
│   └── baseline_provenance.json       # DOI/commit of baseline inputs
│
├── tools/
│   └── import_baseline_metrics.py     # one-time copy/merge utility
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 9) Configuration (YAML)

```yaml
experiment: "graph-tabular-fusion"
seed: 42
device: "cuda"   # or "cpu"

baseline:
  provenance: "docs/baseline_provenance.json"
  metrics_csv: "../elliptic-gnn-baselines/reports/metrics_summary.csv"  # or local copy under tools/
  reuse_splits: true

data:
  root: "data/Elliptic++ Dataset"
  features: "txs_features.csv"
  labels: "txs_classes.csv"
  edges: "txs_edgelist.csv"
  splits_file: "splits.json"
  use_local_only: true      # Protocol A (true) or Protocol B (false)

embed:
  encoder: "node2vec"       # node2vec | graphsage_export | graphsage_train
  out_dim: 64               # 16/32/64/128
  graphsage_ckpt: ""        # path if using export
  save_path: "data/embeddings.parquet"

fusion:
  model: "xgb"              # xgb | mlp
  xgb:
    n_estimators: 800
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    eval_metric: "aucpr"
    early_stopping_rounds: 50

eval:
  recall_k_fracs: [0.005, 0.01, 0.02]
  save_plots: true

logging:
  out_dir: "reports"
```

---

## 10) Metrics & File Formats

**`reports/metrics.json` (per split)**

```json
{
  "pr_auc": 0.8912,
  "roc_auc": 0.9355,
  "best_f1": 0.6420,
  "threshold": 0.421,
  "recall@1%": 0.478
}
```

**Append row to `reports/metrics_summary.csv`:**

```
timestamp,experiment,model,split,pr_auc,roc_auc,f1,recall@1%
1730940200,graph-tabular-fusion,XGB+Node2VecEmb,test,0.891200,0.935500,0.642000,0.478000
```

**Consolidated table**

* `tools/import_baseline_metrics.py` merges baseline CSV rows and fusion rows into one table for README.

---

## 11) Acceptance Criteria (extension milestones)

**M1 — Bootstrap & Import**

* Repo scaffold matches Section 8.
* `pip install -r requirements.txt` succeeds.
* `baseline_provenance.json` created; baseline `metrics_summary.csv` imported/copied.

**M2 — Embeddings MVP**

* `01_generate_embeddings.ipynb` runs (Node2Vec or GraphSAGE export) and saves `data/embeddings.parquet`.

**M3 — Fusion (XGB)**

* `02_fusion_xgb.ipynb` trains fusion model on **Protocol A** and logs metrics + plots.

**M4 — Comparison Report**

* Consolidated table created (baseline vs fusion) and included in README.
* Plots saved to `reports/plots/`.

**M5 — (Optional) Ablations**

* Protocol B and/or embedding size sweep (16/32/64/128).

---

## 12) Risks & Pitfalls (and how we avoid them)

* **Leakage via embeddings:** Compute embeddings **per split** using only within-split edges.
* **Double-encoding neighbor info:** Prefer **Protocol A** by default; document results for Protocol B.
* **ID misalignment:** Persist `txid` mapping; add checks in `merge_embeddings.py`.
* **Baseline mismatch:** Lock baseline DOI/commit in `baseline_provenance.json`; refuse to run if provenance missing.
* **Non-reproducibility:** Seeds + deterministic ops; re-use exact splits; log versions.

---

## 13) Roadmap (future, not here)

* Add Explainability notebook (SHAP for XGB; neighborhood probes).
* Cross-chain replication (Ethereum phishing networks).
* Move to hetero/temporal models in separate repos.

---

## 14) License & Acknowledgements

* Respect Elliptic++ dataset licensing/terms.
* Cite the dataset and the baseline repo’s DOI in README.
* This repo is educational/demonstrative.

---

**End of `PROJECT_SPEC.md` (v2 — Graph–Tabular Fusion Extension).**
