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
from movielens.data import MovieLens

from feature import InputFeatures
from loss import RatingPredictionLoss, Criterion


class Trainer:
    def __init__(self,
                data_name = 'ml-100k',
                valid_ratio = 0.1,
                test_ratio = 0.1,
                inductive=False,
                exp_name=None
                ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inductive = inductive
        self.exp_name = exp_name or f"exp-{time.strftime('%Y%m%d-%H%M%S')}"
        self.dataset = MovieLens(name = data_name,
                                test_ratio = test_ratio,
                                valid_ratio = valid_ratio,
                                device = self.device)

        self.dataset.train_enc_graph = self.dataset.train_enc_graph.int().to(self.device)
        self.dataset.train_dec_graph = self.dataset.train_dec_graph.int().to(self.device)
        self.dataset.valid_enc_graph = self.dataset.train_enc_graph
        self.dataset.valid_dec_graph = self.dataset.valid_dec_graph.int().to(self.device)
        self.dataset.test_enc_graph = self.dataset.test_enc_graph.int().to(self.device)
        self.dataset.test_dec_graph = self.dataset.test_dec_graph.int().to(self.device)

    def train(self,
            n_blocks = 2,
            n_layers_en = 1,
            n_layers_de = 1,
            recurrent = True,
            in_feats_dim = 32,
            e_feats_dim = None,
            en_hidden_feats_dim = 250,
            out_feats_dim = 75,
            r_hidden_feats_dim = 64,
            agg = 'sum',
            drop_out = 0.5,
            activation = 'leaky',
            weight = 0.1,
            p_zero = 0.0,
            p_freeze = 0.0,
            lr = 0.002,
            iteration = 1000,
            log_interval = 1,
            early_stopping = 150,
            lr_intialize_step = 100,
            lr_decay = 0.5,
            train_min_lr = 0.0005
            ):
        wandb.init(
            project="stargcn-movielens",
            name=self.exp_name,
            config={
                "n_blocks": n_blocks,
                "n_layers_en": n_layers_en,
                "n_layers_de": n_layers_de,
                "recurrent": recurrent,
                "in_feats_dim": in_feats_dim,
                "e_feats_dim": e_feats_dim,
                "en_hidden_feats_dim": en_hidden_feats_dim,
                "r_hidden_feats_dim": r_hidden_feats_dim,
                "out_feats_dim": out_feats_dim,
                "agg": agg,
                "drop_out": drop_out,
                "activation": activation,
                "lr": lr, 
                "iteration": iteration,
                "early_stopping": early_stopping,
                "dataset": self.dataset._name if hasattr(self.dataset, '_name') else "unknown"
            }
        )

        n_users, n_items = self.dataset.user_feature.shape[0], self.dataset.movie_feature.shape[0]

        if e_feats_dim is None:
            user_features = InputFeatures(n_nodes = n_users,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2)
            movie_features = InputFeatures(n_nodes = n_items,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2)
        else:
            user_features = InputFeatures(n_nodes = n_users,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2,
                                        e_feats = self.dataset.user_feature,
                                        e_feats_dim = e_feats_dim,
                                        activation = activation)

            movie_features = InputFeatures(n_nodes = n_items,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2,
                                        e_feats = self.dataset.movie_feature,
                                        e_feats_dim = e_feats_dim,
                                        activation = activation)

            in_feats_dim += e_feats_dim

        model = STARGCN(n_blocks = n_blocks,
                        n_layers_en = n_layers_en,
                        n_layers_de = n_layers_de,
                        recurrent = recurrent,
                        edge_types = self.dataset.possible_rating_values,
                        in_feats_dim = in_feats_dim,
                        en_hidden_feats_dim = en_hidden_feats_dim,
                        r_hidden_feats_dim = r_hidden_feats_dim,
                        out_feats_dim = out_feats_dim,
                        agg = agg,
                        drop_out = drop_out,
                        activation = activation)

        print(model)
        print(user_features)
        print(movie_features)

        device = self.device
        model = model.to(device)
        user_features = user_features.to(device)
        movie_features = movie_features.to(device)
        self.e_feats_dim = e_feats_dim

        criterion = Criterion(weight = weight)

        params = list(model.parameters()) + \
            list(user_features.parameters()) + list(movie_features.parameters())
        optimizer = optim.Adam(params, lr = lr)

        train_gt_ratings = self.dataset.train_truths

        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        count_loss = 0

        print(f"Start training on {device}...")
        for iter_idx in range(iteration):
            model.train()

            # TODO : implement inductive version and masked learning
            # ufeats / ifeats: InputFeatures => forward
            ufeats, _, _ = user_features(torch.arange(n_users).to(device))
            ifeats, _, _ = movie_features(torch.arange(n_items).to(device))

            all_ratings, all_recon_feats = \
                model(self.dataset.train_enc_graph, self.dataset.train_dec_graph, ufeats, ifeats)
                    
            loss, rmse = 0., 0.
            for pred_ratings, (ufeats_r, ifeats_r) in zip(all_ratings, all_recon_feats):
                _rmse, _loss = criterion(train_gt_ratings,
                                        pred_ratings,
                                        ufeats,
                                        ifeats,
                                        ufeats_r,
                                        ifeats_r)
                rmse += _rmse
                loss += _loss

            rmse /= len(all_ratings)

            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if iter_idx and iter_idx % log_interval == 0:
                log = f"[{iter_idx}/{iteration}-iter] | [train] loss : {count_loss/iter_idx:.4f}, rmse : {rmse:.4f}"
                wandb.log({
                    "train_loss": count_loss / iter_idx,
                    "train_rmse": rmse
                }, step=iter_idx)
                count_rmse, count_num = 0, 0

            if iter_idx and iter_idx % (log_interval*10) == 0:
                valid_rmse = self.evaluate(model, n_users, n_items, user_features, movie_features, data_type = 'valid')
                wandb.log({"valid_rmse": valid_rmse}, step=iter_idx)
                log += f" | [valid] rmse : {valid_rmse:.4f}"

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    best_test_rmse = self.evaluate(model, n_users, n_items, user_features, movie_features, data_type = 'test')
                    wandb.log({
                        "best_valid_rmse": best_valid_rmse,
                        "best_test_rmse": best_test_rmse
                    }, step=iter_idx)
                    log += f" | [test] rmse : {best_test_rmse:.4f}"

                    torch.save(model, './model.pt')

                else:
                    no_better_valid += 1
                    if no_better_valid > early_stopping:
                        print("Early stopping threshold reached. Stop training.")
                        break
                    if no_better_valid > lr_intialize_step:
                        new_lr = max(lr * lr_decay, train_min_lr)
                        if new_lr < lr:
                            lr = new_lr
                            print("\tChange the LR to %g" % new_lr)
                            for p in optimizer.param_groups:
                                p['lr'] = lr
                            no_better_valid = 0

            if iter_idx and iter_idx  % log_interval == 0:
                print(log)
        wandb.summary["best_valid_rmse"] = best_valid_rmse
        wandb.summary["best_test_rmse"] = best_test_rmse
        wandb.finish()

        print(f'[END] Best Iter : {best_iter} Best Valid RMSE : {best_valid_rmse:.4f}, Best Test RMSE : {best_test_rmse:.4f}')
    def evaluate(self, model, n_users, n_items, user_features, movie_features, data_type = 'valid'):
        if data_type == "valid":
            gt_ratings = self.dataset.valid_truths
            dec_graph = self.dataset.valid_dec_graph
        elif data_type == "test":
            gt_ratings = self.dataset.test_truths
            dec_graph = self.dataset.test_dec_graph

        model.eval()
        with torch.no_grad():
            get_rmse = RatingPredictionLoss()

            if self.inductive:
                if not self.e_feats_dim is None:
                    # get external features for unseen nodes
                    ufeats = user_features.get_unseen_feature(idx=torch.arange(n_users).to(self.device), e_feats=self.dataset.user_feature)
                    ifeats = movie_features.get_unseen_feature(idx=torch.arange(n_items).to(self.device), e_feats=self.dataset.movie_feature)
                else:
                    ufeats = user_features.get_unseen_feature(torch.arange(n_users).to(self.device))
                    ifeats = movie_features.get_unseen_feature(torch.arange(n_items).to(self.device))
            else:
                ufeats, _, _ = user_features(torch.arange(n_users).to(self.device))
                ifeats, _, _ = movie_features(torch.arange(n_items).to(self.device))

            all_ratings, _ = model(self.dataset.train_enc_graph, dec_graph, ufeats, ifeats)
            rmse = 0.
            for pred_ratings in all_ratings:
                rmse += get_rmse(gt_ratings, pred_ratings).pow(1/2)

            rmse /= len(all_ratings)
            
        return rmse

if __name__ == '__main__':
    SEED = 152
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    fire.Fire(Trainer)
