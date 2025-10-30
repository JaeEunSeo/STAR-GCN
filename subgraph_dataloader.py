"""
DataLoader for subgraph-based link prediction
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from subgraph_extractor import extract_enclosing_subgraph
from drnl_computer import compute_drnl_labels


class SubgraphDataset(Dataset):
    """
    Dataset that generates subgraphs for each user-item link
    
    For each (user, item, rating) triple:
    1. Extract h-hop enclosing subgraph
    2. Compute DRNL labels
    3. Return subgraph, features, and rating
    """
    def __init__(self, graph, user_item_pairs, ratings, n_users, n_items, 
                 h=2, max_nodes_per_hop=None, max_label=100):
        """
        Parameters
        ----------
        graph : dgl.DGLGraph
            Full bipartite graph
        user_item_pairs : tuple or list
            (user_indices, item_indices) or list of (user, item) tuples
        ratings : np.ndarray or torch.Tensor
            Rating values for each pair
        n_users : int
            Total number of users
        n_items : int
            Total number of items
        h : int
            Number of hops for subgraph extraction
        max_nodes_per_hop : int or None
            Max nodes to sample per hop
        max_label : int
            Maximum DRNL label value
        """
        self.graph = graph
        self.n_users = n_users
        self.n_items = n_items
        self.h = h
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_label = max_label
        
        # Convert to list of pairs
        if isinstance(user_item_pairs, tuple):
            self.pairs = list(zip(user_item_pairs[0], user_item_pairs[1]))
        else:
            self.pairs = user_item_pairs
        
        # Convert ratings to numpy
        if isinstance(ratings, torch.Tensor):
            self.ratings = ratings.cpu().numpy()
        else:
            self.ratings = np.array(ratings)
        
        assert len(self.pairs) == len(self.ratings), \
            f"Mismatch: {len(self.pairs)} pairs vs {len(self.ratings)} ratings"
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns
        -------
        subgraph : dgl.DGLGraph
            Extracted subgraph
        user_labels : np.ndarray
            DRNL labels for user nodes
        item_labels : np.ndarray
            DRNL labels for item nodes
        rating : float
            Ground truth rating
        node_mapping : dict
            Mapping from subgraph IDs to original IDs
        """
        user_idx, item_idx = self.pairs[idx]
        rating = self.ratings[idx]
        
        # Extract subgraph
        subgraph, node_mapping, target_pos = extract_enclosing_subgraph(
            self.graph,
            user_idx,
            item_idx,
            self.n_users,
            self.n_items,
            h=self.h,
            max_nodes_per_hop=self.max_nodes_per_hop
        )
        
        # Compute DRNL labels
        user_labels, item_labels, _ = compute_drnl_labels(
            subgraph,
            target_user_idx=target_pos[0],
            target_item_idx=target_pos[1],
            max_label=self.max_label
        )
        
        return subgraph, user_labels, item_labels, rating, node_mapping


def collate_subgraphs(batch):
    """
    Collate function for batching subgraphs
    
    Parameters
    ----------
    batch : list of tuples
        [(subgraph, user_labels, item_labels, rating, node_mapping), ...]
    
    Returns
    -------
    batched_subgraphs : list
        List of subgraphs (not batched into one graph, processed separately)
    user_labels_list : list
        List of user label arrays
    item_labels_list : list
        List of item label arrays
    ratings : torch.Tensor
        Batched ratings [batch_size]
    node_mappings : list
        List of node mappings
    """
    subgraphs = []
    user_labels_list = []
    item_labels_list = []
    ratings = []
    node_mappings = []
    
    for subgraph, user_labels, item_labels, rating, node_mapping in batch:
        subgraphs.append(subgraph)
        user_labels_list.append(user_labels)
        item_labels_list.append(item_labels)
        ratings.append(rating)
        node_mappings.append(node_mapping)
    
    ratings_tensor = torch.tensor(ratings, dtype=torch.float32)
    
    return subgraphs, user_labels_list, item_labels_list, ratings_tensor, node_mappings


def create_subgraph_dataloader(graph, user_item_pairs, ratings, n_users, n_items,
                               batch_size=32, h=2, max_nodes_per_hop=None, 
                               max_label=100, shuffle=True, num_workers=0):
    """
    Create DataLoader for subgraph-based training
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        Full bipartite graph
    user_item_pairs : tuple
        (user_indices, item_indices)
    ratings : np.ndarray or torch.Tensor
        Rating values
    n_users : int
        Number of users
    n_items : int
        Number of items
    batch_size : int
        Batch size
    h : int
        Hops for subgraph
    max_nodes_per_hop : int or None
        Max nodes per hop
    max_label : int
        Max DRNL label
    shuffle : bool
        Shuffle data
    num_workers : int
        Number of workers for data loading
    
    Returns
    -------
    dataloader : DataLoader
    """
    dataset = SubgraphDataset(
        graph=graph,
        user_item_pairs=user_item_pairs,
        ratings=ratings,
        n_users=n_users,
        n_items=n_items,
        h=h,
        max_nodes_per_hop=max_nodes_per_hop,
        max_label=max_label
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_subgraphs,
        num_workers=num_workers
    )
    
    return dataloader
