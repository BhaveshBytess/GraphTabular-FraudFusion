"""
E2: Generate Node2Vec Embeddings (Leakage-Free)

This script generates Node2Vec embeddings per temporal split to prevent leakage.
"""

import sys
import torch
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.elliptic_loader import EllipticDataset
from src.embeddings.node2vec import generate_node2vec_embeddings
from src.utils.seed import set_all_seeds

def main():
    print("=" * 60)
    print("E2: Node2Vec Embedding Generation (Leakage-Free)")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "embed_node2vec.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_all_seeds(config['seed'])
    print(f"\n✅ Config loaded: {config['experiment']}")
    print(f"   Seed: {config['seed']}")
    print(f"   Device: {config['device']}")
    
    # Check device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"\n🖥️  Using device: {device}")
    
    # Load dataset
    print("\n📂 Loading Elliptic++ dataset...")
    data_root = Path(__file__).parent.parent / config['data']['root']
    dataset = EllipticDataset(str(data_root), use_local_only=True)
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Total nodes: {len(dataset.features_df)}")
    print(f"   Total edges: {dataset.edge_index.shape[1]}")
    print(f"   Features: {dataset.features.shape[1]}")
    
    # Display split info
    print(f"\n📊 Split distribution:")
    print(f"   Train: {dataset.splits['train'].sum():,} nodes")
    print(f"   Val:   {dataset.splits['val'].sum():,} nodes")
    print(f"   Test:  {dataset.splits['test'].sum():,} nodes")
    
    # Node2Vec config
    n2v_config = config['node2vec']
    train_config = config['training']
    
    print(f"\n⚙️  Node2Vec configuration:")
    print(f"   Embedding dim: {n2v_config['embedding_dim']}")
    print(f"   Walk length: {n2v_config['walk_length']}")
    print(f"   Context size: {n2v_config['context_size']}")
    print(f"   Walks per node: {n2v_config['walks_per_node']}")
    print(f"   p: {n2v_config['p']}, q: {n2v_config['q']}")
    print(f"   Epochs: {train_config['epochs']}")
    
    # Generate embeddings per split (prevent leakage)
    all_embeddings = []
    all_indices = []
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'=' * 60}")
        print(f"Generating embeddings for {split_name.upper()} split")
        print(f"{'=' * 60}")
        
        # Get split data
        split_mask = dataset.splits[split_name]
        features, labels, edge_index_split = dataset.get_split_data(split_name)
        
        print(f"📊 Split info:")
        print(f"   Nodes: {len(features):,}")
        print(f"   Edges: {edge_index_split.shape[1]:,}")
        print(f"   Labeled: {(labels != -1).sum():,}")
        
        # Convert to torch
        edge_index_t = torch.from_numpy(edge_index_split).long()
        
        print(f"\n🔄 Training Node2Vec...")
        
        # Generate embeddings
        embeddings = generate_node2vec_embeddings(
            edge_index=edge_index_t,
            num_nodes=len(features),
            embedding_dim=n2v_config['embedding_dim'],
            walk_length=n2v_config['walk_length'],
            context_size=n2v_config['context_size'],
            walks_per_node=n2v_config['walks_per_node'],
            num_negative_samples=n2v_config['num_negative_samples'],
            p=n2v_config['p'],
            q=n2v_config['q'],
            batch_size=train_config['batch_size'],
            lr=train_config['lr'],
            epochs=train_config['epochs'],
            device=device,
            sparse=n2v_config['sparse']
        )
        
        print(f"✅ Embeddings generated: {embeddings.shape}")
        
        # Store embeddings and their original indices
        split_indices = np.where(split_mask)[0]
        all_embeddings.append(embeddings)
        all_indices.extend(split_indices)
    
    # Combine all embeddings in original order
    print(f"\n{'=' * 60}")
    print("Combining embeddings from all splits...")
    print(f"{'=' * 60}")
    
    # Create full embedding matrix
    full_embeddings = np.zeros((len(dataset.features_df), n2v_config['embedding_dim']), dtype=np.float32)
    
    # Place embeddings at correct indices
    start_idx = 0
    for split_name, embeddings in zip(['train', 'val', 'test'], all_embeddings):
        split_mask = dataset.splits[split_name]
        split_indices = np.where(split_mask)[0]
        full_embeddings[split_indices] = embeddings
        start_idx += len(embeddings)
    
    print(f"✅ Combined embeddings: {full_embeddings.shape}")
    
    # Create DataFrame with txId
    embeddings_df = pd.DataFrame(
        full_embeddings,
        columns=[f'emb_{i}' for i in range(n2v_config['embedding_dim'])]
    )
    embeddings_df.insert(0, 'txId', dataset.features_df['txId'].values)
    
    # Save embeddings
    output_path = Path(__file__).parent.parent / config['output']['save_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving embeddings...")
    embeddings_df.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Embeddings saved to: {output_path}")
    print(f"   Shape: {embeddings_df.shape}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    # Verification
    print(f"\n🔍 Verification:")
    print(f"   Unique txIds: {embeddings_df['txId'].nunique()}")
    print(f"   Expected: {len(dataset.features_df)}")
    print(f"   Match: {embeddings_df['txId'].nunique() == len(dataset.features_df)}")
    
    # Sample check
    print(f"\n📊 Sample embeddings (first 3 nodes):")
    print(embeddings_df.head(3))
    
    print(f"\n{'=' * 60}")
    print("✅ E2 COMPLETE: Embeddings generated successfully!")
    print(f"{'=' * 60}")
    print(f"\n📝 Next step: E3 - Train fusion model (XGBoost)")
    print(f"   Run: notebooks/02_fusion_xgb.ipynb")

if __name__ == "__main__":
    main()
