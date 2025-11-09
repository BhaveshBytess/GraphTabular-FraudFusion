🪩 This runtime prompt assumes the workspace was cloned using:
`CLONE_INIT_PROMPT.md` — setup and provenance verified.


# 🚀 START PROMPT — “Elliptic++ Fraud Detection: Graph–Tabular Fusion (Extension)” Boot

**Context load:**
You are initializing work on the **`graph-tabular-fusion`** repository — a **fusion-only extension** of the completed baseline project **`elliptic-gnn-baselines`**. This repo focuses on **concatenating graph embeddings with tabular features** and evaluating the **marginal value of graph structure** under strict, leakage-free temporal splits.

* **Baseline repo (source of truth):** `elliptic-gnn-baselines` — **COMPLETE** (M1–M10), Zenodo DOI published.
* **This repo (extension):** **fusion-only**, **no retraining** of old baselines or GNNs. We **reuse** the baseline’s `splits.json` and `metrics_summary.csv`.

Your full operational context is defined by three documents in this repo:

1. `docs/AGENT.MD` — **behavioral discipline**, verification rules, escalation protocol.
2. `docs/PROJECT_SPEC.md` — **architecture** for fusion, datasets, metrics, acceptance criteria.
3. `TASKS.md` — **active planner** for milestones/tasks in this extension.

---

## 🧠 Initialization Instructions

1. **Read** `docs/AGENT.MD`, `docs/PROJECT_SPEC.md`, `TASKS.md`.
2. Adopt **Plan → Verify → Execute → Log** from `AGENT.MD`.
3. Treat:

   * `PROJECT_SPEC.md` as **immutable blueprint** for fusion scope.
   * `TASKS.md` as **dynamic plan** (statuses `[ ]`, `[~]`, `[x]`, `[?]`).
4. Confirm dataset path **`data/Elliptic++ Dataset/`** exists with **real Elliptic++** files only:

   * `txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv`, `splits.json`.
5. Confirm **provenance**: `docs/baseline_provenance.json` records baseline URL/commit/DOI.

**Critical constraint:** **Do NOT** retrain legacy baselines (LR/XGB/MLP) or GNNs (GCN/GraphSAGE/GAT). We **reuse** their metrics and the **exact same** temporal splits.

---

## 📈 Current State Snapshot (as of Nov 9, 2025)

**Baseline project (`elliptic-gnn-baselines`)**

* **Status:** ✅ **COMPLETE** (M1–M10)
* **Key finding:** Tabular features (esp. aggregates AF94–AF182) largely encode neighbor info → **XGBoost > GraphSAGE**.
* **Zenodo DOI:** 10.5281/zenodo.17560930
* **Artifacts:** `splits.json`, `reports/metrics_summary.csv`, published docs and plots.

**This extension (`graph-tabular-fusion`)**

* **Status:** 🟡 **INIT** (scaffold + imports to be completed)
* **Goal:** Train **fusion** models (`[tabular || embeddings] → XGBoost/MLP`) with **no leakage** and compare against **imported** baseline metrics.

---

## 🎯 Research Goal (Extension Scope)

> **Primary Objective:** Quantify the **incremental benefit** of graph embeddings when combined with **tabular features**, under the **same temporal splits** as the baseline, and **without** retraining legacy baselines/GNNs.

---

## 🔬 Fusion Design Principles

* **Protocol A (default, safer):** **Local features (AF1–AF93)** + **Embeddings** — avoids double-encoding neighbor statistics.
* **Protocol B (comparative):** **Local+Aggregates (AF1–AF182)** + **Embeddings** — document gains/redundancy.
* **Embeddings (choose one for MVP):**

  1. **Node2Vec** (fast, unsupervised), or
  2. **GraphSAGE export** from baseline checkpoint (penultimate layer).
* **Leakage rule:** Compute embeddings **per split** using **only edges within** that split.

---

## 🧾 Workflow Discipline (unchanged)

1. **Plan** intent + expected outputs.
2. **Verify** dataset presence, columns, split alignment, and leakage constraints.
3. **Execute** minimal, reproducible notebooks/scripts.
4. **Log** metrics to `reports/`, update `TASKS.md`, and save plots.

Escalate after **≤5** failed fix attempts with a succinct failure memo and options.

---

## 🧩 Extension Milestones (v2)

| Milestone                       | Goal                                                                                                                   | Deliverables                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **E1 — Bootstrap & Provenance** | Scaffold repo; import baseline **`splits.json`** and **`metrics_summary.csv`**; write `docs/baseline_provenance.json`. | Folder tree, imported CSV, provenance JSON.                                  |
| **E2 — Embeddings (MVP)**       | Produce **leakage-safe** embeddings (Node2Vec **or** GraphSAGE export).                                                | `data/embeddings.parquet` (with `txid` + columns).                           |
| **E3 — Fusion Train (XGBoost)** | Build **Protocol A** fused features and train/evaluate **XGBoost** with early stopping on **val PR-AUC**.              | `reports/metrics.json`, updated `reports/metrics_summary.csv`, PR/ROC plots. |
| **E4 — Side-by-Side Report**    | Merge **imported baseline** rows with **fusion** rows; produce a single comparison table and brief README summary.     | Consolidated table in README; `reports/plots/*.png`.                         |
| **E5 — (Optional) Ablations**   | Protocol B and/or embedding size sweep (16/32/64/128).                                                                 | Added rows + small bar chart.                                                |

---

## 📊 Evaluation Protocol (unchanged from spec)

* **Primary metric:** PR-AUC; also report ROC-AUC, F1 (threshold from **val**), Recall@K (0.5%, 1%, 2%).
* **Artifacts:** `reports/metrics.json`, append to `reports/metrics_summary.csv`, plots saved to `reports/plots/`.

---

## 🧭 Behavioral Highlights (from AGENT.MD)

* **No synthetic data.**
* **No retraining** of baseline models; **reuse** their metrics & splits.
* **Explain before executing.**
* **Leakage gate:** If embeddings include cross-split edges, **stop** and request correction.
* **Sanity gate:** If PR-AUC seems implausible (>0.90), trigger a **LeakageSuspect** review before acceptance.

---

## ✅ Start Command (for new chat)

1. Summarize `PROJECT_SPEC.md` and `TASKS.md` (fusion scope only).
2. Confirm presence of:

   * `data/Elliptic++ Dataset/{txs_features.csv, txs_classes.csv, txs_edgelist.csv, splits.json}`
   * `docs/baseline_provenance.json`
   * Imported baseline `reports/metrics_summary.csv`
3. Choose embedding path (Node2Vec **or** GraphSAGE export) based on availability/speed.
4. Build **Protocol A** fused dataset and train **XGBoost**.
5. Produce **side-by-side** table in README (baseline vs fusion).

---

### 🪩 Output Expectation for New Sessions

* Clear restatement of fusion scope and constraints.
* Confirmation that **baseline artifacts** (splits + metrics) are in place.
* Confirmation of **leakage-safe** embedding plan.
* Fusion results logged and merged with baseline table.
* README updated with a short, honest comparison summary.

---

## 📦 Key Inputs & Paths

* **Dataset folder:** `data/Elliptic++ Dataset/`
* **Baseline imports:** `data/Elliptic++ Dataset/splits.json`, `reports/metrics_summary.csv`
* **Embeddings output:** `data/embeddings.parquet`
* **Fusion artifact:** `data/fused.parquet`
* **Reports:** `reports/metrics.json`, `reports/plots/*.png`, `reports/metrics_summary.csv`
* **Provenance:** `docs/baseline_provenance.json`

---

## 🏁 Status Policy

* **E1–E4 complete** → Fusion MVP accepted.
* **E5 optional** → Ablation extras.
* Any violation of leakage/repro rules → **Stop**, summarize, and request a decision.

---

**End of Start Prompt (v2 — Graph–Tabular Fusion Extension, updated 2025-11-09)**

---
