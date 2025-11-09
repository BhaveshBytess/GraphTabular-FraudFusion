"""
E3: Train XGBoost Fusion Model

Merge embeddings with tabular features and train XGBoost classifier.
"""

import sys
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.elliptic_loader import EllipticDataset
from src.data.merge_embeddings import merge_embeddings_with_features
from src.train.fusion_xgb import train_xgb_fusion
from src.eval.fusion_report import create_comparison_report
from src.utils.seed import set_all_seeds

def main():
    print("=" * 60)
    print("E3: XGBoost Fusion Model Training")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "fusion_xgb.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_all_seeds(config['seed'])
    print(f"\n✅ Config loaded: {config['experiment']}")
    print(f"   Seed: {config['seed']}")
    print(f"   Protocol: {'A (Local-only)' if config['data']['use_local_only'] else 'B (All features)'}")
    
    # Load dataset
    print("\n📂 Loading Elliptic++ dataset...")
    data_root = Path(__file__).parent.parent / config['data']['root']
    dataset = EllipticDataset(str(data_root), use_local_only=config['data']['use_local_only'])
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Total nodes: {len(dataset.features_df):,}")
    print(f"   Features: {dataset.features.shape[1]}")
    print(f"   Train: {dataset.splits['train'].sum():,}")
    print(f"   Val: {dataset.splits['val'].sum():,}")
    print(f"   Test: {dataset.splits['test'].sum():,}")
    
    # Merge embeddings with features
    print(f"\n🔗 Merging embeddings with features...")
    embeddings_path = Path(__file__).parent.parent / config['embed']['save_path']
    features_path = data_root / config['data']['features']
    
    fused_df = merge_embeddings_with_features(
        str(embeddings_path),
        str(features_path),
        use_local_only=config['data']['use_local_only'],
        output_path=None
    )
    
    print(f"\n✅ Fusion complete:")
    print(f"   Total features: {len(fused_df.columns) - 2}")  # Exclude txId and Time step
    print(f"   Tabular: {dataset.features.shape[1]}")
    print(f"   Embeddings: {config['embed']['out_dim']}")
    
    # Prepare train/val/test splits
    print(f"\n📊 Preparing splits...")
    
    # Create a mapping from txId to row index in fused_df
    fused_df['_row_idx'] = range(len(fused_df))
    txid_to_row = dict(zip(fused_df['txId'], fused_df['_row_idx']))
    
    # Get masks from dataset (these are boolean arrays for the original dataset order)
    train_mask = dataset.splits['train']
    val_mask = dataset.splits['val']
    test_mask = dataset.splits['test']
    
    # Get labeled nodes only (exclude unknown class -1)
    train_labeled = dataset.get_labeled_mask(train_mask)
    val_labeled = dataset.get_labeled_mask(val_mask)
    test_labeled = dataset.get_labeled_mask(test_mask)
    
    # Get txIds for labeled nodes in each split
    train_txids = dataset.features_df.loc[train_labeled, 'txId'].values
    val_txids = dataset.features_df.loc[val_labeled, 'txId'].values
    test_txids = dataset.features_df.loc[test_labeled, 'txId'].values
    
    # Map txIds to row indices in fused_df
    train_indices = [txid_to_row[txid] for txid in train_txids if txid in txid_to_row]
    val_indices = [txid_to_row[txid] for txid in val_txids if txid in txid_to_row]
    test_indices = [txid_to_row[txid] for txid in test_txids if txid in txid_to_row]
    
    # Extract feature columns (exclude txId, Time step, _row_idx)
    feature_cols = [col for col in fused_df.columns if col not in ['txId', 'Time step', '_row_idx']]
    
    X_train = fused_df.iloc[train_indices][feature_cols].values
    y_train = dataset.labels[train_labeled]
    
    X_val = fused_df.iloc[val_indices][feature_cols].values
    y_val = dataset.labels[val_labeled]
    
    X_test = fused_df.iloc[test_indices][feature_cols].values
    y_test = dataset.labels[test_labeled]
    
    print(f"\n✅ Splits prepared:")
    print(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    
    # Class distribution
    print(f"\n📊 Class distribution:")
    print(f"   Train - Fraud: {(y_train == 1).sum():,} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")
    print(f"   Train - Legit: {(y_train == 0).sum():,} ({(y_train == 0).sum() / len(y_train) * 100:.2f}%)")
    print(f"   Val   - Fraud: {(y_val == 1).sum():,} ({(y_val == 1).sum() / len(y_val) * 100:.2f}%)")
    print(f"   Test  - Fraud: {(y_test == 1).sum():,} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")
    
    # Train XGBoost
    print(f"\n{'=' * 60}")
    print("Training XGBoost Fusion Model")
    print(f"{'=' * 60}")
    
    output_dir = Path(__file__).parent.parent / config['logging']['out_dir']
    
    xgb_config = config['fusion']['xgb'].copy()
    xgb_config['device'] = config['device']
    
    model, metrics = train_xgb_fusion(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        config=xgb_config,
        output_dir=str(output_dir)
    )
    
    # Display results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    
    print(f"\n📊 Validation Metrics:")
    for k, v in metrics['val'].items():
        print(f"   {k}: {v:.4f}")
    
    print(f"\n📊 Test Metrics:")
    for k, v in metrics['test'].items():
        print(f"   {k}: {v:.4f}")
    
    # Generate comparison report
    print(f"\n{'=' * 60}")
    print("Generating Comparison Report")
    print(f"{'=' * 60}")
    
    baseline_csv = Path(__file__).parent.parent / config['baseline']['metrics_csv']
    
    comparison_df = create_comparison_report(
        fusion_metrics=metrics,
        baseline_csv=str(baseline_csv),
        output_dir=str(output_dir)
    )
    
    # Show comparison
    print(f"\n📊 Model Comparison (Test Set):")
    test_comparison = comparison_df[comparison_df['split'] == 'test'][['model', 'pr_auc', 'roc_auc', 'f1']]
    print(test_comparison.sort_values('pr_auc', ascending=False).to_string(index=False))
    
    print(f"\n{'=' * 60}")
    print("✅ E3 COMPLETE: Fusion model trained successfully!")
    print(f"{'=' * 60}")
    print(f"\n📝 Outputs saved:")
    print(f"   - Model: {output_dir}/xgb_fusion.json")
    print(f"   - Metrics: {output_dir}/metrics.json")
    print(f"   - Comparison: {output_dir}/metrics_summary.csv")
    print(f"   - Plots: {output_dir}/plots/")
    
    print(f"\n📝 Next step: E4 - Update README with comparison results")

if __name__ == "__main__":
    main()
