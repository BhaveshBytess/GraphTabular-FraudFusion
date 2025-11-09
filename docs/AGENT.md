# 🧭 AGENTv2.MD — Operational Discipline for Codex Agent

## 🎯 Project Context

**Project:** `Graph–Tabular Hybrid Model` (GNN–XGB Fusion)  
**Status:** 🧩 **IN PROGRESS** (Phase II of Elliptic++ Study — extending baselines)  
**Goal:** Build, train, and evaluate **hybrid models** that fuse **graph embeddings** (from GraphSAGE/GCN) with **tabular ML learners** (XGBoost, MLP) on the **Elliptic++** dataset.  
**Purpose:** Extend the baseline GNN work into practical, interpretable models — combining relational learning with tabular performance for real-world fraud detection use cases.  

---

### Agent Mode Toggle
To enable flexible behavior across repositories, the agent can operate in different modes:

| Mode | Description | Typical Use |
|------|--------------|--------------|
| `RESEARCH` | Enables full verification, logging, and dataset reproducibility checks. | Default mode for active experiments and ablations. |
| `MAINTENANCE` | Reduces verbosity; skips heavy verification for quick bugfixes. | Routine updates or post-publication fixes. |
| `EXPERIMENTAL` | Allows architectural freedom — dynamic fusion, feature ablation, or integration testing. | Prototyping hybrid Graph–Tabular models. |
| `ANALYTICS` | Evaluation-only mode — runs saved models and aggregates metrics. | Comparative analysis and publication reporting. |

Set with environment variable or config flag:
```bash
export AGENT_MODE=RESEARCH
````

---

## 🧠 Core Philosophy

**Rule:** *Think before you code.*
Every action follows this discipline:

> **Plan → Verify → Execute → Log**

1. **Plan**: Explain what you intend to do (in comments or markdown).
2. **Verify**: Check dataset availability, paths, imports, and prior outputs.
3. **Execute**: Run only when context and inputs are validated.
4. **Log**: Record metrics, plots, and notes; never just say “done”.

---

## 📝 To-Do List Discipline

Use an explicit, living TODO checklist to drive every action.
Never start a new task until the current task’s checklist is ✅ complete.

**Rules**

* Maintain a single project checklist in `TASKS.md` and a mini-checklist at the top of each notebook.
* Every task has: **ID, Goal, Steps, Done criteria** (must include verification).
* Update the checklist before and after each operation:

  **Before:** mark planned steps as pending and state expected outputs.
  **After:** mark completed steps, attach artifact paths, and note warnings/errors.

If blocked > 5 fix attempts → stop, write an escalation note (what was tried, errors, hypotheses), and request guidance.

**Allowed statuses**

```
[ ] pending
[~] in progress
[?] blocked (requires input)
[x] done (after verification)
```

### Project-level template (TASKS.md)

```
# TASKS (single source of truth)

## T-01 Bootstrap Repo
Goal: Scaffold folders, README, requirements, configs.
Steps:
- [ ] Create folder tree and empty notebooks
- [ ] Add requirements.txt and install
- [ ] Add configs (fusion_xgb, fusion_mlp, embed_graphsage)
Done when:
- [x] pip install -r requirements.txt succeeds
- [x] Tree matches scaffold; README renders

## T-02 Embeddings + Fusion Pipeline
Goal: Train GraphSAGE embeddings, export features, and fuse with tabular data.
Steps:
- [ ] Train GraphSAGE and export node embeddings (.parquet)
- [ ] Merge embeddings with Elliptic++ tabular features
- [ ] Train baseline XGBoost / MLP models on fused data
- [ ] Log metrics and plots under reports/
Done when:
- [x] reports/metrics_summary.csv updated
- [x] Fusion model reproducible with same seed
```

### Notebook-level header template

```
# Notebook TODO (auto-discipline)
- [ ] Load real Elliptic++ data from data/elliptic/
- [ ] Generate or load GraphSAGE embeddings
- [ ] Fuse embeddings + tabular features
- [ ] Train fusion model (XGBoost/MLP)
- [ ] Save: reports/metrics.json, plots/, append metrics_summary.csv
- [ ] Verify metrics + artifact paths printed in last cell
- [ ] Clear TODOs before commit
```

**Execution protocol with TODOs**

Plan → expand TASKS.md + header checklist → Verify paths/config → Execute steps sequentially → Log artifacts.
Blocked? mark `[?]` and add Escalation Note before asking.

---

## ⚙️ Decision Chain Discipline

The agent must never assume. It must reason and confirm.

1. Describe intended change and expected outcome.
2. Validate environment (paths, packages, variables).
3. Run minimal safe code to verify correctness.
4. Summarize results and check warnings/errors.
5. If uncertain → pause and ask.

**Forbidden behaviors:**
– Blind continuation after exceptions
– Skipping error resolution
– Fabricating synthetic data

---

## 🧩 Data Handling Rules

**Dataset Identity:**
`Elliptic++` — real Bitcoin transaction dataset with graph + tabular structure.

**Data Policy:**

* 📁 Data lives in `data/Elliptic++ Dataset/`:

  * `txs_features.csv` — 182 tabular features
  * `txs_classes.csv` — labels (1=Fraud, 2=Legit, 3=Unknown)
  * `txs_edgelist.csv` — graph connections
* 🛑 Never fabricate or mock data.
* 🧾 Always verify file existence before import.
* 📥 Download location: [Google Drive Folder](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l)
* 💾 All metrics and plots must reference the dataset version in use.

---

## 📓 Notebook Workflow Discipline

**Main work happens in notebooks under `/notebooks`.**

### Notebook Rules

1. Each experiment (fusion, ablation, explainability) in its own `.ipynb`.
2. Use markdown cells for objectives + findings.
3. Keep code concise and readable.
4. `/src` for reusable utilities (loaders, metrics, fusion modules).
5. Each notebook should:

   * Load data
   * Run one experiment
   * Produce:

     * `reports/metrics.json`
     * `reports/plots/*.png`
     * Append `metrics_summary.csv`

### Notebook Flow Example

| Step | Notebook                       | Purpose                      |
| ---- | ------------------------------ | ---------------------------- |
| 0    | `00_baselines_tabular.ipynb`   | XGBoost / MLP baselines      |
| 1    | `01_generate_embeddings.ipynb` | GraphSAGE → embedding export |
| 2    | `02_fusion_xgb.ipynb`          | Tabular + embedding fusion   |
| 3    | `03_ablation_studies.ipynb`    | Fusion feature ablations     |
| 4    | `04_explainability.ipynb`      | SHAP / GNNExplainer insights |

---

## 🧮 Verification Before Commit

Before declaring any task **complete**, verify:

✅ All notebooks run end-to-end on real Elliptic++ data.
✅ Metrics logged (`metrics_summary.csv`, `metrics.json`).
✅ PR-AUC / ROC-AUC / Recall@K plotted and saved.
✅ No TODOs or placeholders remain.
✅ All paths relative (`data/elliptic/...`).
✅ Seeds set (torch, numpy, python).
✅ No hardcoded absolute paths or env leaks.

---

## 🧰 Error & Resolution Protocol

If an error occurs:

1. **Stop immediately.**
2. Attempt fix ≤ 5 times with reasoning.
3. For each attempt log: what / why / result.
4. If unresolved → summarize causes, notify user, await decision.

Never continue “as if it worked.”

---

## 📊 Logging & Artifact Discipline

Every run outputs:

* `reports/metrics_summary.csv` — all experiment results
* `reports/plots/*.png` — PR/ROC curves & fusion visuals
* `checkpoints/model_best.pt` (if applicable)
* `data/elliptic/splits.json` — temporal splits

Each `metrics_summary.csv` row must include:

| Field      | Example              |
| ---------- | -------------------- |
| timestamp  | 1730940200           |
| experiment | graph-tabular-fusion |
| model      | XGB + GraphSAGE emb  |
| split      | test                 |
| pr_auc     | 0.8912               |
| roc_auc    | 0.9355               |
| f1         | 0.642                |
| recall@1%  | 0.478                |

---

## 🧬 Reproducibility

Always call:

```python
from src.utils.seed import set_all_seeds
set_all_seeds(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

before any training step.

Log:

* Python / PyTorch / PyG versions
* Random seeds in JSON configs

---

## 🧑‍💻 Communication Tone & Escalation

**Tone:** Analytical, cautious, transparent.
Always explain *why* before *doing*.

If progress stalls or data errors persist:

```
❗ Stopped execution
Attempted fixes:
 1. …
 2. …
Remaining issue: …
Possible causes: …
Awaiting instruction.
```

Never hide or skip failed cells.

---

## ✅ Summary

| Aspect            | Policy                                            |
| ----------------- | ------------------------------------------------- |
| Dataset           | Real Elliptic++ only                              |
| Code surface      | Primarily notebooks                               |
| Verification      | Strict & reproducible                             |
| Decision protocol | Plan → Verify → Execute → Log                     |
| Errors            | Resolve or escalate                               |
| Communication     | Transparent & reasoned                            |
| Goal              | A robust, interpretable Graph–Tabular Fusion repo |

---

**End of AGENT.MD**

---

