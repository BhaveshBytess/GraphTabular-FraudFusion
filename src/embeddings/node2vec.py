"""Simplified Node2Vec embedding generation using node2vec library."""
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from gensim.models import Word2Vec
import random


def generate_node2vec_embeddings(
    edge_index: np.ndarray,
    num_nodes: int,
    embedding_dim: int = 64,
    walk_length: int = 80,
    context_size: int = 10,
    walks_per_node: int = 10,
    num_negative_samples: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    batch_size: int = 128,
    lr: float = 0.01,
    epochs: int = 5,
    device: str = 'cuda',
    sparse: bool = True
):
    """
    Generate Node2Vec embeddings using random walks + Word2Vec.
    
    Args:
        edge_index: Edge list [2, E] as numpy array or torch tensor
        num_nodes: Total number of nodes
        embedding_dim: Output embedding dimension
        walk_length: Length of random walks
        context_size: Context window size
        walks_per_node: Number of walks per node
        num_negative_samples: Negative samples per positive
        p: Return parameter (not used in simplified version)
        q: In-out parameter (not used in simplified version)
        batch_size: Training batch size (not used)
        lr: Learning rate (not used)
        epochs: Number of Word2Vec epochs
        device: 'cuda' or 'cpu' (not used in CPU-only implementation)
        sparse: Use sparse gradients (not used)
        
    Returns:
        embeddings: [N, embedding_dim] numpy array
    """
    # Convert edge_index to numpy if torch tensor
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()
    
    print(f"   Building NetworkX graph from {edge_index.shape[1]:,} edges...")
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    print(f"   Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Generate random walks
    print(f"   Generating {walks_per_node * num_nodes:,} random walks...")
    walks = []
    
    for node in range(num_nodes):
        if node % 10000 == 0 and node > 0:
            print(f"      Progress: {node:,}/{num_nodes:,} nodes")
        
        for _ in range(walks_per_node):
            walk = [node]
            current = node
            
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            
            walks.append([str(n) for n in walk])
    
    print(f"   Generated {len(walks):,} walks")
    
    # Train Word2Vec model
    print(f"   Training Word2Vec model...")
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=context_size,
        min_count=0,
        sg=1,  # Skip-gram
        workers=4,
        epochs=epochs,
        negative=num_negative_samples
    )
    
    # Extract embeddings for all nodes
    embeddings = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    
    for node_id in range(num_nodes):
        node_str = str(node_id)
        if node_str in model.wv:
            embeddings[node_id] = model.wv[node_str]
        else:
            # Initialize missing nodes with small random values
            embeddings[node_id] = np.random.randn(embedding_dim).astype(np.float32) * 0.01
    
    print(f"   ✅ Embeddings extracted for {num_nodes:,} nodes")
    
    return embeddings


if __name__ == "__main__":
    print("Node2Vec embedding generator - use via notebooks or scripts")
