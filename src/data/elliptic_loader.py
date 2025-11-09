"""Elliptic++ dataset loader with temporal splits."""
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from .splits import create_temporal_splits, filter_edges_by_split


class EllipticDataset:
    """Loader for Elliptic++ dataset with temporal splits."""
    
    def __init__(self, root: str, use_local_only: bool = True):
        """
        Initialize Elliptic++ dataset.
        
        Args:
            root: Path to 'Elliptic++ Dataset' folder
            use_local_only: If True, use only Local features (Protocol A)
        """
        self.root = Path(root)
        self.use_local_only = use_local_only
        
        # Load data
        self.features_df = pd.read_csv(self.root / "txs_features.csv")
        self.labels_df = pd.read_csv(self.root / "txs_classes.csv")
        self.edges_df = pd.read_csv(self.root / "txs_edgelist.csv")
        
        # Create txid mapping
        self.txid_to_idx = {txid: idx for idx, txid in enumerate(self.features_df['txId'])}
        self.idx_to_txid = {idx: txid for txid, idx in self.txid_to_idx.items()}
        
        # Process features
        self._process_features()
        
        # Process labels (1=fraud, 2=legit, 3=unknown -> 1, 0, -1)
        self.labels = self._process_labels()
        
        # Process edges
        self.edge_index = self._process_edges()
        
        # Create temporal splits
        timestamps = self.features_df['Time step'].values
        self.splits = create_temporal_splits(timestamps)
        
        print(f"✅ Loaded Elliptic++ dataset")
        print(f"   Nodes: {len(self.features_df)}")
        print(f"   Edges: {self.edge_index.shape[1]}")
        print(f"   Features: {self.features.shape[1]}")
        print(f"   Train: {self.splits['train'].sum()}, Val: {self.splits['val'].sum()}, Test: {self.splits['test'].sum()}")
    
    def _process_features(self):
        """Extract and normalize features."""
        if self.use_local_only:
            # Protocol A: Local features only (AF1-93)
            feature_cols = [col for col in self.features_df.columns if col.startswith('Local_feature_')]
        else:
            # Protocol B: All features (AF1-182)
            feature_cols = [col for col in self.features_df.columns 
                          if col.startswith('Local_feature_') or col.startswith('Aggregate_feature_')]
        
        self.features = self.features_df[feature_cols].values.astype(np.float32)
    
    def _process_labels(self):
        """Convert labels to binary format."""
        # Merge labels with features
        merged = self.features_df[['txId']].merge(self.labels_df, on='txId', how='left')
        labels = merged['class'].fillna(3).astype(int).values
        
        # Convert: 1 (fraud) -> 1, 2 (legit) -> 0, 3 (unknown) -> -1
        binary_labels = np.where(labels == 1, 1, np.where(labels == 2, 0, -1))
        return binary_labels
    
    def _process_edges(self):
        """Convert edge list to tensor format."""
        # Map txids to indices
        src_indices = self.edges_df['txId1'].map(self.txid_to_idx).dropna().astype(int)
        dst_indices = self.edges_df['txId2'].map(self.txid_to_idx).dropna().astype(int)
        
        # Stack to [2, E] format
        edge_index = np.stack([src_indices.values, dst_indices.values], axis=0)
        return edge_index
    
    def get_split_data(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific split with leakage-free edges.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            features, labels, edge_index (filtered to split)
        """
        mask = self.splits[split]
        
        # Filter edges to only include nodes in this split
        edge_index_split = filter_edges_by_split(self.edge_index, mask)
        
        return self.features[mask], self.labels[mask], edge_index_split
    
    def get_labeled_mask(self, split_mask: np.ndarray = None) -> np.ndarray:
        """Get mask for labeled nodes (excludes unknown class -1)."""
        labeled_mask = self.labels != -1
        if split_mask is not None:
            labeled_mask = labeled_mask & split_mask
        return labeled_mask


if __name__ == "__main__":
    # Test loader
    dataset = EllipticDataset("data/Elliptic++ Dataset", use_local_only=True)
    print(f"Dataset loaded successfully")
