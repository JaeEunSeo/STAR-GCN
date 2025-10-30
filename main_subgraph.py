"""
SEAL + STAR-GCN Hybrid Training
Subgraph-based link prediction with recurrent reconstruction
"""
import sys
sys.path.append('..')

import os
import time
import fire
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import STARGCN
from movielens.data import MovieLens, to_etype_name

from feature import DRNLSubgraphFeatures
from loss import RatingPredictionLoss, Criterion
from subgraph_dataloader import create_subgraph_dataloader


class SubgraphTrainer:
    def __init__(self,
                data_name='ml-100k',
                valid_ratio=0.1,
                test_ratio=0.1,
                exp_name=None):
        """
        SEAL + STAR-GCN Hybrid Trainer
        
        Uses:
        - SEAL's subgraph extraction and DRNL labeling
        - STAR-GCN's recurrent reconstruction on each subgraph
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.exp_name = exp_name or f"subgraph-exp-{time.strftime('%Y%m%d-%H%M%S')}"
        
        print(f"Loading {data_name} dataset...")
        self.dataset = MovieLens(
            name=data_name,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            device=self.device
        )
        
        self.n_users = self.dataset.num_user
        self.n_items = self.dataset.num_movie
        
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Train: {len(self.dataset.train_truths)}, "
              f"Valid: {len(self.dataset.valid_truths)}, "
              f"Test: {len(self.dataset.test_truths)}")

    def train(self,
            # SEAL subgraph parameters
            h=2,
            max_nodes_per_hop=None,
            drnl_max_label=100,
            batch_size=32,
            
            # STAR-GCN model parameters  
            n_blocks=2,
            n_layers_en=1,
            n_layers_de=1,
            recurrent=True,
            in_feats_dim=32,
            en_hidden_feats_dim=250,
            out_feats_dim=75,
            r_hidden_feats_dim=64,
            agg='sum',
            drop_out=0.5,
            activation='leaky',
            weight=0.1,
            
            # Training parameters
            lr=0.002,
            iteration=100,  # epochs
            log_interval=1,
            early_stopping=20,
            lr_decay=0.5,
            train_min_lr=0.0005
            ):
        """
        Train STAR-GCN on subgraphs with DRNL features
        """
        wandb.init(
            project="stargcn-seal-hybrid",
            name=self.exp_name,
            config={
                "h": h,
                "max_nodes_per_hop": max_nodes_per_hop,
                "drnl_max_label": drnl_max_label,
                "batch_size": batch_size,
                "n_blocks": n_blocks,
                "n_layers_en": n_layers_en,
                "n_layers_de": n_layers_de,
                "recurrent": recurrent,
                "in_feats_dim": in_feats_dim,
                "en_hidden_feats_dim": en_hidden_feats_dim,
                "out_feats_dim": out_feats_dim,
                "r_hidden_feats_dim": r_hidden_feats_dim,
                "agg": agg,
                "drop_out": drop_out,
                "activation": activation,
                "lr": lr,
                "iteration": iteration,
                "dataset": self.dataset._name
            }
        )
        
        # Create DRNL feature module
        print("Creating DRNL feature module...")
        drnl_features = DRNLSubgraphFeatures(
            max_label=drnl_max_label,
            emb_dim=in_feats_dim
        ).to(self.device)
        
        print(f"Original train_enc_graph edge types: {self.dataset.train_enc_graph.canonical_etypes}")
        
        # Convert rating values to simple numeric edge type names (e.g., 1.0 -> '1')
        edge_type_names = [str(int(r)) for r in self.dataset.possible_rating_values]
        print(f"Edge type names for model: {edge_type_names}")
        
        # Create STAR-GCN model
        print("Creating STAR-GCN model...")
        model = STARGCN(
            n_blocks=n_blocks,
            n_layers_en=n_layers_en,
            n_layers_de=n_layers_de,
            recurrent=recurrent,
            edge_types=edge_type_names,  # Use converted edge type names
            in_feats_dim=in_feats_dim,
            en_hidden_feats_dim=en_hidden_feats_dim,
            out_feats_dim=out_feats_dim,
            r_hidden_feats_dim=r_hidden_feats_dim,
            agg=agg,
            drop_out=drop_out,
            activation=activation
        ).to(self.device)
        
        # Create dataloaders
        print("Creating subgraph dataloaders...")
        
        # Training pairs
        train_pairs = (
            self.dataset.train_dec_graph.edges(etype='rate')[0].cpu().numpy(),
            self.dataset.train_dec_graph.edges(etype='rate')[1].cpu().numpy()
        )
        
        train_loader = create_subgraph_dataloader(
            graph=self.dataset.train_enc_graph,
            user_item_pairs=train_pairs,
            ratings=self.dataset.train_truths.cpu().numpy(),
            n_users=self.n_users,
            n_items=self.n_items,
            batch_size=batch_size,
            h=h,
            max_nodes_per_hop=max_nodes_per_hop,
            max_label=drnl_max_label,
            shuffle=True
        )
        
        # Validation pairs
        valid_pairs = (
            self.dataset.valid_dec_graph.edges(etype='rate')[0].cpu().numpy(),
            self.dataset.valid_dec_graph.edges(etype='rate')[1].cpu().numpy()
        )
        
        valid_loader = create_subgraph_dataloader(
            graph=self.dataset.train_enc_graph,  # Note: use train_enc for valid too
            user_item_pairs=valid_pairs,
            ratings=self.dataset.valid_truths.cpu().numpy(),
            n_users=self.n_users,
            n_items=self.n_items,
            batch_size=batch_size,
            h=h,
            max_nodes_per_hop=max_nodes_per_hop,
            max_label=drnl_max_label,
            shuffle=False
        )
        
        # Test pairs
        test_pairs = (
            self.dataset.test_dec_graph.edges(etype='rate')[0].cpu().numpy(),
            self.dataset.test_dec_graph.edges(etype='rate')[1].cpu().numpy()
        )
        
        test_loader = create_subgraph_dataloader(
            graph=self.dataset.test_enc_graph,
            user_item_pairs=test_pairs,
            ratings=self.dataset.test_truths.cpu().numpy(),
            n_users=self.n_users,
            n_items=self.n_items,
            batch_size=batch_size,
            h=h,
            max_nodes_per_hop=max_nodes_per_hop,
            max_label=drnl_max_label,
            shuffle=False
        )
        
        # Optimizer
        params = list(model.parameters()) + list(drnl_features.parameters())
        optimizer = optim.Adam(params, lr=lr)
        
        # Loss
        criterion = Criterion(weight=weight)
        rmse_loss = RatingPredictionLoss()
        
        # Training loop
        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(iteration):
            model.train()
            drnl_features.train()
            
            epoch_loss = 0.0
            epoch_rmse = 0.0
            batch_count = 0
             
            # mini-batch training
            for subgraphs, user_labels_list, item_labels_list, ratings, _ in train_loader:
                batch_loss = 0.0
                batch_rmse = 0.0
                
                ratings = ratings.to(self.device)
                
                # Process each subgraph in the batch
                for subgraph, user_labels, item_labels, rating in zip(
                    subgraphs, user_labels_list, item_labels_list, ratings
                ):
                    subgraph = subgraph.to(self.device)
                    rating_target = rating.unsqueeze(0)
                    
                    # Debug: Check subgraph edge types (only for first batch)
                    if epoch == 0 and batch_count == 0:
                        print(f"Subgraph edge types: {subgraph.canonical_etypes}")
                    
                    # Convert DRNL labels to features
                    user_feats = drnl_features(user_labels)
                    item_feats = drnl_features(item_labels)
                    
                    # Forward pass through STAR-GCN on this subgraph
                    all_ratings, all_recon_feats = model(
                        subgraph, subgraph,  # enc_graph, dec_graph
                        user_feats, item_feats
                    )
                    
                    # Get prediction for target link (position 0, 0)
                    # The target link is the first user and first item in subgraph
                    pred_rating = all_ratings[-1][0]  # Last block, target position
                    
                    # Compute loss
                    rmse = torch.sqrt(rmse_loss(rating_target, pred_rating.unsqueeze(0)))
                    
                    # Reconstruction loss
                    loss = 0.0
                    for pred_ratings, (ufeats_r, ifeats_r) in zip(all_ratings, all_recon_feats):
                        _, _loss = criterion(
                            rating_target,
                            pred_ratings[0].unsqueeze(0).unsqueeze(0),
                            user_feats,
                            item_feats,
                            ufeats_r,
                            ifeats_r
                        )
                        loss += _loss
                    
                    batch_loss += loss
                    batch_rmse += rmse.item()
                
                # Average over batch
                batch_loss /= len(subgraphs)
                batch_rmse /= len(subgraphs)
                
                # Backward
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                epoch_rmse += batch_rmse
                batch_count += 1
            
            # Average over epoch
            epoch_loss /= batch_count
            epoch_rmse /= batch_count
            
            if epoch % log_interval == 0:
                log = f"[{epoch}/{iteration}] | train loss: {epoch_loss:.4f}, rmse: {epoch_rmse:.4f}"
                
                wandb.log({
                    "train_loss": epoch_loss,
                    "train_rmse": epoch_rmse
                }, step=epoch)
                
                # Validation
                if epoch % (log_interval * 5) == 0:
                    valid_rmse = self.evaluate(
                        model, drnl_features, valid_loader, "valid"
                    )
                    wandb.log({"valid_rmse": valid_rmse}, step=epoch)
                    log += f" | valid rmse: {valid_rmse:.4f}"
                    
                    if valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        no_better_valid = 0
                        best_iter = epoch
                        
                        # Test
                        best_test_rmse = self.evaluate(
                            model, drnl_features, test_loader, "test"
                        )
                        wandb.log({
                            "best_valid_rmse": best_valid_rmse,
                            "best_test_rmse": best_test_rmse
                        }, step=epoch)
                        log += f" | test rmse: {best_test_rmse:.4f}"
                        
                        torch.save(model.state_dict(), './model_subgraph.pt')
                    else:
                        no_better_valid += 1
                        if no_better_valid > early_stopping:
                            print("Early stopping!")
                            break
                        if no_better_valid > 5:
                            new_lr = max(lr * lr_decay, train_min_lr)
                            if new_lr < lr:
                                lr = new_lr
                                print(f"\tLR → {new_lr}")
                                for p in optimizer.param_groups:
                                    p['lr'] = lr
                                no_better_valid = 0
                
                print(log)
        
        wandb.summary["best_valid_rmse"] = best_valid_rmse
        wandb.summary["best_test_rmse"] = best_test_rmse
        wandb.finish()
        
        print(f"\n[END] Best Epoch: {best_iter} | "
              f"Valid RMSE: {best_valid_rmse:.4f} | "
              f"Test RMSE: {best_test_rmse:.4f}")
    
    def evaluate(self, model, drnl_features, dataloader, split_name):
        """Evaluate on validation or test set"""
        model.eval()
        drnl_features.eval()
        
        rmse_loss = RatingPredictionLoss()
        total_rmse = 0.0
        total_count = 0
        
        with torch.no_grad():
            for subgraphs, user_labels_list, item_labels_list, ratings, _ in dataloader:
                ratings = ratings.to(self.device)
                
                for subgraph, user_labels, item_labels, rating in zip(
                    subgraphs, user_labels_list, item_labels_list, ratings
                ):
                    subgraph = subgraph.to(self.device)
                    rating_target = rating.unsqueeze(0)
                    
                    # Features
                    user_feats = drnl_features(user_labels)
                    item_feats = drnl_features(item_labels)
                    
                    # Forward
                    all_ratings, _ = model(subgraph, subgraph, user_feats, item_feats)
                    
                    # Prediction for target link
                    pred_rating = all_ratings[-1][0, 0]
                    
                    # RMSE
                    rmse = torch.sqrt(rmse_loss(rating_target, pred_rating.unsqueeze(0)))
                    total_rmse += rmse.item()
                    total_count += 1
        
        avg_rmse = total_rmse / total_count
        return avg_rmse


if __name__ == '__main__':
    SEED = 152
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    fire.Fire(SubgraphTrainer)
