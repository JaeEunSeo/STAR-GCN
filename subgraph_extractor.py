"""
SEAL-style subgraph extraction for link prediction
Adapted from SEAL/Python/util_functions.py
"""
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
from collections import defaultdict
from utils import add_degree



def extract_enclosing_subgraph(graph, user_idx, item_idx, n_users, n_items, h=2, max_nodes_per_hop=None):
    """
    Extract h-hop enclosing subgraph around a target link (user_idx, item_idx)
    
    This follows SEAL's approach:
    1. Find all h-hop neighbors of both user and item
    2. Extract the induced subgraph
    3. Keep the bipartite structure (user nodes on one side, item nodes on other)
    
    Parameters
    ----------
    graph : dgl.DGLGraph
        The full bipartite graph (user-item)
    user_idx : int
        Target user node ID in the full graph
    item_idx : int
        Target item node ID in the full graph  
    n_users : int
        Total number of users in full graph
    n_items : int
        Total number of items in full graph
    h : int
        Number of hops for enclosing subgraph
    max_nodes_per_hop : int or None
        Maximum nodes to sample per hop (for large graphs)
    
    Returns
    -------
    subgraph : dgl.DGLGraph
        The extracted bipartite subgraph
    node_mapping : dict
        Mapping from subgraph node IDs to original graph IDs
        {'user': {subgraph_id: original_id}, 'item': {subgraph_id: original_id}}
    target_in_subgraph : tuple
        (user_subgraph_id, item_subgraph_id) - position of target link in subgraph
    """
    # Convert to adjacency matrix for easier neighbor finding
    adj_matrix = graph_to_adj_matrix(graph, n_users, n_items)
    
    # Extract h-hop neighbors
    user_nodes, item_nodes = h_hop_neighbors(
        adj_matrix, user_idx, item_idx, n_users, h, max_nodes_per_hop
    )
    
    # Make sure target nodes are included and put them first
    if user_idx not in user_nodes:
        user_nodes.add(user_idx)
    if item_idx not in item_nodes:
        item_nodes.add(item_idx)
    
    # Convert to sorted lists (target nodes first)
    user_nodes.discard(user_idx)
    item_nodes.discard(item_idx)
    user_nodes_list = [user_idx] + sorted(list(user_nodes))
    item_nodes_list = [item_idx] + sorted(list(item_nodes))
    
    # Create node mappings
    user_map = {i: orig_id for i, orig_id in enumerate(user_nodes_list)}
    item_map = {i: orig_id for i, orig_id in enumerate(item_nodes_list)}
    user_inv_map = {orig_id: i for i, orig_id in enumerate(user_nodes_list)}
    item_inv_map = {orig_id: i for i, orig_id in enumerate(item_nodes_list)}
    
    # Extract subgraph edges
    subgraph_edges = extract_subgraph_edges(
        adj_matrix, user_nodes_list, item_nodes_list, user_inv_map, item_inv_map
    )
    
    # Build DGL bipartite subgraph
    subgraph = build_dgl_bipartite_graph(
        subgraph_edges, len(user_nodes_list), len(item_nodes_list)
    )
    
    node_mapping = {
        'user': user_map,
        'item': item_map
    }
    
    # Target link is at (0, 0) in subgraph
    target_in_subgraph = (0, 0)
    
    return subgraph, node_mapping, target_in_subgraph


def graph_to_adj_matrix(graph, n_users, n_items):
    """Convert DGL heterograph to scipy sparse adjacency matrix with rating values"""
    # Build user-item adjacency matrix with rating values
    row, col, data = [], [], []
    
    # Iterate over all edge types and extract rating from edge type name
    for etype in graph.canonical_etypes:
        src_type, edge_type, dst_type = etype
        if 'user' in src_type and 'item' in dst_type and not edge_type.startswith('reverse'):
            src, dst = graph.edges(etype=etype)
            
            # Extract rating value from edge type (e.g., '5_0' -> 5.0)
            try:
                rating_str = edge_type.replace('_', '.')
                rating = float(rating_str)
            except:
                rating = 1.0  # default
            
            row.extend(src.cpu().numpy())
            col.extend(dst.cpu().numpy())
            data.extend([rating] * len(src))
    
    adj = ssp.csr_matrix((data, (row, col)), shape=(n_users, n_items))
    return adj


def h_hop_neighbors(adj_matrix, user_idx, item_idx, n_users, h, max_nodes_per_hop=None):
    """
    Find all h-hop neighbors of the target link
    
    Returns
    -------
    user_nodes : set
        Set of user node IDs
    item_nodes : set
        Set of item node IDs
    """
    # Build full adjacency for BFS
    # Structure: [users, items]
    n_items = adj_matrix.shape[1]
    full_adj = ssp.bmat([
        [None, adj_matrix],
        [adj_matrix.T, None]
    ], format='csr')
    
    # Target nodes in full graph
    user_node_id = user_idx
    item_node_id = n_users + item_idx
    
    # BFS from both target nodes
    visited = set([user_node_id, item_node_id])
    current_layer = set([user_node_id, item_node_id])
    
    for hop in range(h):
        next_layer = set()
        for node in current_layer:
            # Find neighbors
            neighbors = full_adj[node].nonzero()[1]
            
            # Sample if too many neighbors
            if max_nodes_per_hop and len(neighbors) > max_nodes_per_hop:
                neighbors = np.random.choice(
                    neighbors, size=max_nodes_per_hop, replace=False
                )
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_layer.add(neighbor)
                    visited.add(neighbor)
        
        current_layer = next_layer
        if len(current_layer) == 0:
            break
    
    # Split into user and item nodes
    user_nodes = set([n for n in visited if n < n_users])
    item_nodes = set([n - n_users for n in visited if n >= n_users])
    
    return user_nodes, item_nodes


def extract_subgraph_edges(adj_matrix, user_nodes, item_nodes, user_inv_map, item_inv_map):
    """Extract edges within the subgraph with rating values"""
    edges = []
    
    for u_orig in user_nodes:
        for i_orig in item_nodes:
            if adj_matrix[u_orig, i_orig] > 0:
                u_sub = user_inv_map[u_orig]
                i_sub = item_inv_map[i_orig]
                # Store rating value as well
                rating = adj_matrix[u_orig, i_orig]
                edges.append((u_sub, i_sub, rating))
    
    return edges


def build_dgl_bipartite_graph(edges, n_user_nodes, n_item_nodes, possible_ratings=None):
    """
    Build DGL bipartite graph from edge list
    
    Creates multiple edge types based on rating values to match MovieLens format
    IMPORTANT: Always create ALL rating edge types (even if empty) so STAR-GCN can process it
    """
    if possible_ratings is None:
        # Default ratings for MovieLens - MUST match training graph
        possible_ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Group edges by rating
    rating_edges = {r: ([], []) for r in possible_ratings}
    
    for edge in edges:
        if len(edge) == 3:
            u_sub, i_sub, rating = edge
        else:
            u_sub, i_sub = edge
            rating = 5.0  # default
        
        # Round rating to nearest possible rating
        rating = min(possible_ratings, key=lambda x: abs(x - rating))
        rating_edges[rating][0].append(u_sub)
        rating_edges[rating][1].append(i_sub)
    
    # Build graph data dict with ALL rating-specific edge types
    # Use simple numeric string format: '1', '2', '3', '4', '5'
    # Include both forward and reverse edges for degree calculation
    graph_data = {}
    for rating in possible_ratings:
        rating_str = str(int(rating))  # Use '1', '2', '3', '4', '5' format
        
        if len(rating_edges[rating][0]) > 0:
            src = torch.tensor(rating_edges[rating][0], dtype=torch.long)
            dst = torch.tensor(rating_edges[rating][1], dtype=torch.long)
        else:
            # IMPORTANT: Create empty edge type (not None)
            src = torch.tensor([], dtype=torch.long)
            dst = torch.tensor([], dtype=torch.long)
        
        # Add both forward and reverse edge types (needed for degree calculation)
        graph_data[('user', rating_str, 'item')] = (src, dst)
        graph_data[('item', f'reverse-{rating_str}', 'user')] = (dst, src)
    
    subgraph = dgl.heterograph(
        graph_data,
        num_nodes_dict={'user': n_user_nodes, 'item': n_item_nodes}
    )
    
    # Add degree information (ci, cj) required by GCMC
    # Use simple numeric edge type names: ['1', '2', '3', '4', '5']
    edge_type_names = [str(int(r)) for r in possible_ratings]
    add_degree(subgraph, edge_types=edge_type_names)
    
    return subgraph


def remove_target_link(subgraph, target_user_id=0, target_item_id=0):
    """
    Remove the target link from subgraph (used for DRNL computation)
    
    In SEAL, we compute node labels with the target link removed,
    so the GNN learns to predict whether the link should exist.
    Works with multiple rating edge types.
    """
    graph_data = {}
    
    # Process all edge types
    for etype in subgraph.canonical_etypes:
        src_type, edge_type, dst_type = etype
        
        if 'user' in src_type and 'item' in dst_type:
            # Get edges for this rating type
            src, dst = subgraph.edges(etype=etype)
            
            # Find edges that are NOT the target link
            mask = ~((src == target_user_id) & (dst == target_item_id))
            
            # Keep only non-target edges
            new_src = src[mask]
            new_dst = dst[mask]
            
            graph_data[(src_type, edge_type, dst_type)] = (new_src, new_dst)
        
        elif 'item' in src_type and 'user' in dst_type:
            # Reverse edges
            src, dst = subgraph.edges(etype=etype)
            
            # Find edges that are NOT the target link (reversed)
            mask = ~((dst == target_user_id) & (src == target_item_id))
            
            new_src = src[mask]
            new_dst = dst[mask]
            
            graph_data[(src_type, edge_type, dst_type)] = (new_src, new_dst)
    
    new_subgraph = dgl.heterograph(
        graph_data,
        num_nodes_dict={
            'user': subgraph.num_nodes('user'),
            'item': subgraph.num_nodes('item')
        }
    )
    
    return new_subgraph
