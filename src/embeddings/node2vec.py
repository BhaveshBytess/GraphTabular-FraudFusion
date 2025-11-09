"""Node2Vec embedding generation for Elliptic++."""
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path


def generate_node2vec_embeddings(
    edge_index: torch.Tensor,
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
    Generate Node2Vec embeddings.
    
    Args:
        edge_index: Edge list [2, E]
        num_nodes: Total number of nodes
        embedding_dim: Output embedding dimension
        walk_length: Length of random walks
        context_size: Context window size
        walks_per_node: Number of walks per node
        num_negative_samples: Negative samples per positive
        p: Return parameter
        q: In-out parameter
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        sparse: Use sparse gradients
        
    Returns:
        embeddings: [N, embedding_dim] numpy array
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize Node2Vec model
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=sparse
    ).to(device)
    
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model().cpu().numpy()
    
    return embeddings


if __name__ == "__main__":
    print("Node2Vec embedding generator - use via notebooks or scripts")
