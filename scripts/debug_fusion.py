"""Debug fusion model performance issue."""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("Debugging Fusion Model Performance")
print("=" * 60)

# Load embeddings
print("\n1. Checking embeddings...")
emb = pd.read_parquet('data/embeddings.parquet')
print(f"   Shape: {emb.shape}")
print(f"   txId range: {emb['txId'].min()} to {emb['txId'].max()}")

emb_cols = [c for c in emb.columns if c.startswith('emb_')]
print(f"   Embedding stats:")
print(f"      Mean: {emb[emb_cols].mean().mean():.4f}")
print(f"      Std: {emb[emb_cols].std().mean():.4f}")
print(f"      Any NaN: {emb[emb_cols].isna().any().any()}")
print(f"      Any Inf: {np.isinf(emb[emb_cols]).any().any()}")

# Load features
print("\n2. Checking features...")
feat = pd.read_csv('data/Elliptic++ Dataset/txs_features.csv')
print(f"   Shape: {feat.shape}")
print(f"   txId range: {feat['txId'].min()} to {feat['txId'].max()}")

# Check alignment
print("\n3. Checking alignment...")
merged = feat.merge(emb, on='txId', how='inner')
print(f"   Merged shape: {merged.shape}")
print(f"   All features matched: {len(merged) == len(feat)}")

# Load labels
print("\n4. Checking labels...")
labels = pd.read_csv('data/Elliptic++ Dataset/txs_classes.csv')
print(f"   Shape: {labels.shape}")
print(f"   Class distribution:")
print(labels['class'].value_counts().sort_index())

# Check metrics file
print("\n5. Checking saved metrics...")
import json
with open('reports/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"   Val PR-AUC: {metrics['val']['pr_auc']:.4f}")
print(f"   Test PR-AUC: {metrics['test']['pr_auc']:.4f}")

# Compare with baseline
print("\n6. Comparing with baseline...")
baseline_df = pd.read_csv('reports/metrics_summary.csv')
baseline_xgb = baseline_df[baseline_df['model'] == 'XGBoost'].sort_values('pr_auc', ascending=False).head(1)
print(f"   Best baseline XGBoost PR-AUC: {baseline_xgb['pr_auc'].values[0]:.4f}")
print(f"   Fusion PR-AUC: {metrics['test']['pr_auc']:.4f}")
print(f"   Difference: {baseline_xgb['pr_auc'].values[0] - metrics['test']['pr_auc']:.4f}")

print("\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
