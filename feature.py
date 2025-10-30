import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as ssp
import networkx as nx
from scipy.sparse.csgraph import shortest_path

from utils import activation_map


class ExternalFeatures(nn.Module):
    def __init__(self,
                feats,
                out_feats_dim,
                activation
                ):
        super().__init__()
        """Two layer feedforward NN for external feature transformation

        Parameters
        ----------
        feats : torch.FloatTensor
            external features
        iout_feats_dim : int
            dimension of output feature
        activation : str
            activation type
        """
        self.feats = feats

        self.W_1 = nn.Linear(feats.shape[1], out_feats_dim)
        self.activation = activation_map[activation]
        self.W_2 = nn.Linear(out_feats_dim, out_feats_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W_1.weight)
        torch.nn.init.xavier_normal_(self.W_2.weight)

    def transform(self, feats):
        feats = self.W_1(feats)
        feats = self.activation(feats)
        feats = self.W_2(feats)

        return feats

    def forward(self, idx):
        """
        Parameters
        ----------
        idx : list or torch.LongTensor
            target indices
        
        Returns
        -------
        feats : torch.Tensor
            W_{2} \sigma( W_{1} @ feats)
        """
        feats = self.transform(self.feats[idx])

        return feats

class InputFeatures(nn.Module):
    def __init__(self,
                n_nodes,
                emb_dim,
                p_zero = 0.2,
                p_freeze = 0.,
                e_feats = None,
                e_feats_dim = None,
                activation = None
                ):
        super().__init__()
        """STAR-GCN input features

        Parameters
        ----------
        n_nodes : int
            number of nodes
        emb_dim : int
            embedding size
        """
        self.p_zero = p_zero
        self.p_freeze = p_freeze

        if e_feats is None:
            self.external_feats = None
        else:
            self.external_feats = ExternalFeatures(feats = e_feats,
                                                    out_feats_dim = e_feats_dim,
                                                    activation = activation)
        self.emb_dim = emb_dim
        print(f"emb_dim: {emb_dim}. n_nodes: {n_nodes}")
        self.feats = nn.Embedding(n_nodes, emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.feats.weight)

    def get_unseen_feature(self, idx, e_feats = None):
        """generate unseen node features to inductive inference

        Parameters
        ----------
        idx: torch.LongTensor
        e_feats : torch.FloatTensor (optional)
            externel features
        """
        device = self.feats.weight.device

        feats = torch.zeros(len(idx), self.emb_dim).to(device)
        if e_feats is not None:
            efeats = self.external_feats.transform(e_feats.to(device))
            feats = torch.cat([feats, efeats], dim = -1)

        return feats

    def generate_mask(self, idx, p):
        length, device = len(idx), idx.device
        if p <= 0.:
            mask = None
        else:
            mask = (torch.rand(length, ) < p).to(device)

        return mask

    def forward(self, idx, masked = True):
        '''
        return: feats, mask_zero, mask_freeze
            feats: torch.Tensor node feature 
            mask_zero: boolean idx mask for zero-masking
            mask_freeze: boolean idx mask for freezing
        '''
        feats = self.feats(idx)

        if masked:
            mask_zero = self.generate_mask(idx, self.p_zero)
            mask_freeze = self.generate_mask(idx, self.p_freeze)
            if mask_zero is not None:
                feats.masked_fill_(mask_zero.unsqueeze(1), 0.)
        else:
            mask_zero, mask_freeze = None, None

        # 노드 외부 피처를 사용하는 경우 -> structural feature + external feature concat
        if self.external_feats is not None:
            external_feats = self.external_feats(idx)
            feats = torch.cat([feats, external_feats], dim = -1)

        return feats, mask_zero, mask_freeze


class DRNLFeatures(nn.Module):
    """Simplified structural features for link prediction
    
    Instead of computing DRNL for each specific link (which is computationally expensive),
    this uses graph structural properties like degree, clustering coefficient, etc.
    as node features, similar to SEAL's approach but adapted for batch processing.
    """
    def __init__(self,
                 graph,
                 n_users,
                 n_items,
                 emb_dim,
                 max_label=10,
                 e_feats_user=None,
                 e_feats_item=None,
                 e_feats_dim=None,
                 activation=None):
        super().__init__()
        """
        Parameters
        ----------
        graph : dgl.DGLGraph
            The bipartite graph structure
        n_users : int
            Number of user nodes
        n_items : int  
            Number of item nodes
        emb_dim : int
            Embedding dimension for structural features
        max_label : int
            Maximum label value for one-hot encoding of degrees
        e_feats_user : torch.Tensor (optional)
            External features for users
        e_feats_item : torch.Tensor (optional)
            External features for items
        e_feats_dim : int (optional)
            Dimension of external features
        activation : str (optional)
            Activation function name
        """
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_label = max_label
        
        # Compute structural features (degree-based)
        self.user_structural_feats = self._compute_structural_features(
            graph, n_users, node_type='user'
        )
        self.item_structural_feats = self._compute_structural_features(
            graph, n_items, node_type='item'
        )
        
        # Feature dimension: degree one-hot (max_label+1) -> project to emb_dim
        self.user_projection = nn.Linear(max_label + 1, emb_dim)
        self.item_projection = nn.Linear(max_label + 1, emb_dim)
        
        # External features
        if e_feats_user is not None:
            self.external_feats_user = ExternalFeatures(
                feats=e_feats_user,
                out_feats_dim=e_feats_dim,
                activation=activation
            )
        else:
            self.external_feats_user = None
            
        if e_feats_item is not None:
            self.external_feats_item = ExternalFeatures(
                feats=e_feats_item,
                out_feats_dim=e_feats_dim,
                activation=activation
            )
        else:
            self.external_feats_item = None
            
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.user_projection.weight)
        torch.nn.init.xavier_normal_(self.item_projection.weight)
    
    def _compute_structural_features(self, graph, n_nodes, node_type='user'):
        """Compute degree-based structural features
        
        Returns one-hot encoded degrees (clipped to max_label)
        """
        try:
            import dgl
            if isinstance(graph, dgl.DGLGraph):
                # Compute out-degrees for the specified node type
                degrees = torch.zeros(n_nodes, dtype=torch.long)
                
                # Sum degrees across all edge types
                for etype in graph.canonical_etypes:
                    if node_type == 'user' and 'user' in etype[0]:
                        src_nodes = graph.edges(etype=etype)[0]
                        for node_id in src_nodes:
                            degrees[node_id] += 1
                    elif node_type == 'item' and 'item' in etype[2]:
                        dst_nodes = graph.edges(etype=etype)[1]
                        for node_id in dst_nodes:
                            degrees[node_id] += 1
                
                # Clip to max_label
                degrees = torch.clamp(degrees, 0, self.max_label)
                
                # One-hot encode
                one_hot = torch.zeros(n_nodes, self.max_label + 1)
                one_hot.scatter_(1, degrees.unsqueeze(1), 1)
                
                return one_hot
        except Exception as e:
            print(f"Warning: Could not compute structural features: {e}")
            # Fallback: return zeros
            return torch.zeros(n_nodes, self.max_label + 1)
    
    def forward(self, idx, node_type='user', target_link=None, masked=False):
        """
        Parameters
        ----------
        idx : torch.LongTensor
            Node indices to get features for
        node_type : str
            'user' or 'item'
        target_link : tuple (optional)
            Unused, kept for interface compatibility with SEAL's DRNL
        masked : bool
            Unused, kept for interface compatibility
            
        Returns
        -------
        feats : torch.Tensor
            Node features
        mask_zero : None
            Kept for interface compatibility
        mask_freeze : None
            Kept for interface compatibility
        """
        device = self.user_projection.weight.device
        
        # Get structural features
        if node_type == 'user':
            structural_feats = self.user_structural_feats[idx].to(device)
            feats = self.user_projection(structural_feats)
        else:  # item
            structural_feats = self.item_structural_feats[idx].to(device)
            feats = self.item_projection(structural_feats)
        
        # Add external features if available
        if node_type == 'user' and self.external_feats_user is not None:
            external_feats = self.external_feats_user(idx)
            feats = torch.cat([feats, external_feats], dim=-1)
        elif node_type == 'item' and self.external_feats_item is not None:
            external_feats = self.external_feats_item(idx)
            feats = torch.cat([feats, external_feats], dim=-1)
        
        return feats, None, None


class DRNLSubgraphFeatures(nn.Module):
    """
    SEAL-style DRNL features for subgraph-based link prediction
    
    This uses the EXACT DRNL computation from SEAL:
    - Each link gets its own subgraph
    - DRNL labels are computed based on distances to target link endpoints
    - Labels are embedded using learnable embedding layer
    """
    def __init__(self, max_label, emb_dim):
        super().__init__()
        """
        Parameters
        ----------
        max_label : int
            Maximum DRNL label value
        emb_dim : int
            Embedding dimension for labels
        """
        self.max_label = max_label
        self.emb_dim = emb_dim
        
        # Learnable label embedding
        self.label_embedding = nn.Embedding(max_label + 1, emb_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.label_embedding.weight)
    
    def forward(self, labels):
        """
        Convert DRNL integer labels to embeddings
        
        Parameters
        ----------
        labels : np.ndarray or torch.LongTensor
            DRNL labels for nodes [n_nodes]
        
        Returns
        -------
        features : torch.Tensor
            Embedded features [n_nodes, emb_dim]
        """
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        
        device = self.label_embedding.weight.device
        labels = labels.to(device)
        
        # Clip to max_label
        labels = torch.clamp(labels, 0, self.max_label)
        
        # Embed
        features = self.label_embedding(labels)
        
        return features