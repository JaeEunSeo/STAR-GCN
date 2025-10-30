"""
SEAL's DRNL (Double Radius Node Labeling)
Exact implementation from SEAL/Python/util_functions.py
"""
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import dgl
import torch


def compute_drnl_labels(subgraph, target_user_idx=0, target_item_idx=0, max_label=1000):
    """
    Compute DRNL (Double Radius Node Labeling) for all nodes in subgraph
    
    This is the EXACT implementation from SEAL paper.
    
    The DRNL label of a node encodes:
    1. Its distance to the first target node (user)
    2. Its distance to the second target node (item)
    3. Combined in a specific formula
    
    Parameters
    ----------
    subgraph : dgl.DGLGraph
        Bipartite subgraph with 'user' and 'item' node types
        Target link should be at positions (0, 0)
    target_user_idx : int
        User node ID in subgraph (usually 0)
    target_item_idx : int
        Item node ID in subgraph (usually 0)
    max_label : int
        Maximum label value (labels beyond this are clipped)
    
    Returns
    -------
    user_labels : np.ndarray
        DRNL labels for all user nodes
    item_labels : np.ndarray
        DRNL labels for all item nodes
    max_label_value : int
        Actual maximum label value in this subgraph
    """
    n_users = subgraph.num_nodes('user')
    n_items = subgraph.num_nodes('item')
    
    # Convert to full adjacency matrix (bipartite)
    adj_matrix = subgraph_to_adj(subgraph, n_users, n_items)
    
    # Build full graph: [users, items] concatenated
    full_adj = ssp.bmat([
        [None, adj_matrix],
        [adj_matrix.T, None]
    ], format='csr')
    
    # Node IDs in full graph
    user_node_id = target_user_idx
    item_node_id = n_users + target_item_idx
    
    # Remove target link (SEAL approach: predict if link should exist)
    full_adj_copy = full_adj.copy()
    full_adj_copy[user_node_id, item_node_id] = 0
    full_adj_copy[item_node_id, user_node_id] = 0
    full_adj_copy.eliminate_zeros()
    
    # Compute shortest paths from both target nodes
    try:
        dist_matrix = shortest_path(
            full_adj_copy,
            directed=False,
            unweighted=True,
            indices=[user_node_id, item_node_id]
        )
    except Exception as e:
        print(f"Warning: shortest_path failed: {e}")
        # Fallback: all nodes get label 0
        user_labels = np.zeros(n_users, dtype=np.int64)
        item_labels = np.zeros(n_items, dtype=np.int64)
        user_labels[target_user_idx] = 1
        item_labels[target_item_idx] = 1
        return user_labels, item_labels, 1
    
    dist_to_user = dist_matrix[0, :]
    dist_to_item = dist_matrix[1, :]
    
    # Apply DRNL formula (from SEAL paper)
    # label = 1 + min(d_u, d_i) + (d//2) * ((d//2) + (d%2) - 1)
    # where d = d_u + d_i
    d = dist_to_user + dist_to_item
    d_over_2 = d // 2
    d_mod_2 = d % 2
    
    labels = 1 + np.minimum(dist_to_user, dist_to_item) + \
             d_over_2 * (d_over_2 + d_mod_2 - 1)
    
    # Handle infinities (disconnected nodes)
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0
    labels[labels < 0] = 0
    
    # Target nodes always get label 1
    labels[user_node_id] = 1
    labels[item_node_id] = 1
    
    # Clip to max_label
    labels = np.minimum(labels, max_label).astype(np.int64)
    
    # Split into user and item labels
    user_labels = labels[:n_users]
    item_labels = labels[n_users:]
    
    max_label_value = int(max(labels.max(), 1))
    
    return user_labels, item_labels, max_label_value


def subgraph_to_adj(subgraph, n_users, n_items):
    """Convert DGL bipartite subgraph to adjacency matrix"""
    # Get edges
    try:
        src, dst = subgraph.edges(etype='rates')
    except:
        # Try other edge type names
        etypes = subgraph.canonical_etypes
        for etype in etypes:
            if 'user' in etype[0] and 'item' in etype[2]:
                src, dst = subgraph.edges(etype=etype)
                break
    
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()
    data = np.ones(len(src))
    
    adj = ssp.csr_matrix((data, (src, dst)), shape=(n_users, n_items))
    return adj


def drnl_labels_to_features(user_labels, item_labels, embedding_dim, device='cpu'):
    """
    Convert DRNL integer labels to embedding features
    
    Parameters
    ----------
    user_labels : np.ndarray
        DRNL labels for users
    item_labels : np.ndarray
        DRNL labels for items
    embedding_dim : int
        Dimension of label embeddings
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    user_features : torch.Tensor
        User node features [n_users, embedding_dim]
    item_features : torch.Tensor
        Item node features [n_items, embedding_dim]
    max_label : int
        Maximum label value (for embedding table size)
    """
    max_label = max(user_labels.max(), item_labels.max())
    
    # Create label embedding
    label_embedding = torch.nn.Embedding(max_label + 1, embedding_dim)
    torch.nn.init.xavier_uniform_(label_embedding.weight)
    label_embedding = label_embedding.to(device)
    
    # Convert labels to tensors
    user_labels_t = torch.from_numpy(user_labels).long().to(device)
    item_labels_t = torch.from_numpy(item_labels).long().to(device)
    
    # Embed
    user_features = label_embedding(user_labels_t)
    item_features = label_embedding(item_labels_t)
    
    return user_features, item_features, max_label


def batch_compute_drnl_features(subgraphs, embedding_dim, max_label=100, device='cpu'):
    """
    Compute DRNL features for a batch of subgraphs
    
    Parameters
    ----------
    subgraphs : list of dgl.DGLGraph
        List of bipartite subgraphs
    embedding_dim : int
        Dimension of label embeddings
    max_label : int
        Maximum label value to use
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    features_list : list of tuples
        [(user_features, item_features), ...]
    global_max_label : int
        Maximum label across all subgraphs
    """
    # Create global label embedding
    label_embedding = torch.nn.Embedding(max_label + 1, embedding_dim)
    torch.nn.init.xavier_uniform_(label_embedding.weight)
    label_embedding = label_embedding.to(device)
    
    features_list = []
    global_max_label = 0
    
    for subgraph in subgraphs:
        # Compute DRNL labels
        user_labels, item_labels, local_max = compute_drnl_labels(
            subgraph, target_user_idx=0, target_item_idx=0, max_label=max_label
        )
        
        global_max_label = max(global_max_label, local_max)
        
        # Convert to tensors
        user_labels_t = torch.from_numpy(user_labels).long().to(device)
        item_labels_t = torch.from_numpy(item_labels).long().to(device)
        
        # Embed
        user_features = label_embedding(user_labels_t)
        item_features = label_embedding(item_labels_t)
        
        features_list.append((user_features, item_features))
    
    return features_list, global_max_label
