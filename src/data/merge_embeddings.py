"""Merge graph embeddings with tabular features."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def merge_embeddings_with_features(
    embeddings_path: str,
    features_path: str,
    use_local_only: bool = True,
    output_path: str = None
) -> pd.DataFrame:
    """
    Left-join embeddings with tabular features on txid.
    
    Args:
        embeddings_path: Path to embeddings file (parquet/npy with txid)
        features_path: Path to txs_features.csv
        use_local_only: If True, use only Local features (AF1-93), else all (AF1-182)
        output_path: Optional path to save merged dataset
        
    Returns:
        DataFrame with [txid, features..., embeddings...]
    """
    # Load embeddings
    if embeddings_path.endswith('.parquet'):
        embeddings_df = pd.read_parquet(embeddings_path)
    elif embeddings_path.endswith('.npy'):
        # Assume accompanying txid file or index alignment
        embeddings = np.load(embeddings_path)
        embeddings_df = pd.DataFrame(embeddings)
        # Add txid if available
        txid_path = embeddings_path.replace('.npy', '_txid.npy')
        if Path(txid_path).exists():
            txids = np.load(txid_path)
            embeddings_df.insert(0, 'txId', txids)
    else:
        raise ValueError(f"Unsupported embedding format: {embeddings_path}")
    
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Select feature subset
    if use_local_only:
        # Protocol A: Local features only (AF1-93)
        feature_cols = ['txId', 'Time step'] + [f'Local_feature_{i}' for i in range(1, 94)]
        available_cols = [col for col in feature_cols if col in features_df.columns]
        features_subset = features_df[available_cols]
    else:
        # Protocol B: All features (AF1-182)
        features_subset = features_df
    
    # Merge on txId
    merged = features_subset.merge(embeddings_df, on='txId', how='inner')
    
    print(f"✅ Merged {len(merged)} nodes")
    print(f"   Features: {len(features_subset.columns) - 1}")
    print(f"   Embeddings: {len(embeddings_df.columns) - 1}")
    print(f"   Total: {len(merged.columns) - 1}")
    
    if output_path:
        merged.to_parquet(output_path, index=False)
        print(f"✅ Saved to {output_path}")
    
    return merged


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python merge_embeddings.py <embeddings_path> <features_path> [--all-features] [--output <path>]")
        sys.exit(1)
    
    emb_path = sys.argv[1]
    feat_path = sys.argv[2]
    use_local = '--all-features' not in sys.argv
    
    output = None
    if '--output' in sys.argv:
        output = sys.argv[sys.argv.index('--output') + 1]
    
    merge_embeddings_with_features(emb_path, feat_path, use_local, output)
