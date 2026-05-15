#!/usr/bin/env python3
"""
Training and evaluation script for HCGATCL-SR.
Uses preprocessed pickle files for train/test splits, builds Data objects,
initializes model, trains with cross-entropy + hard negative contrastive loss,
and evaluates Precision@K and MRR@K.
"""

import argparse
import pickle
import time
from util import Data, split_validation
from model import *        
import os
from tqdm import tqdm

# -------------------- Command-line hyperparameters --------------------
parser = argparse.ArgumentParser(description='HCGATCL-SR runner')
parser.add_argument('--dataset', default='diginetica', help='dataset folder: diginetica / Tmall / Nowplaying')
parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
parser.add_argument('--batchSize', type=int, default=100, help='mini-batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding dimension')
parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization weight')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=1, help='number of GNN layers (int in model)')
parser.add_argument('--beta', type=float, default=0.01, help='weight for contrastive SSL loss')
parser.add_argument('--filter', type=bool, default=False, help='whether to filter very small values in adjacency')
parser.add_argument('--cut_data', type=int, default=0, help='use first N sessions for quick debug')
parser.add_argument('--hard_k', type=int, default=10, help='number of hard negative samples')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature for contrastive loss')
opt = parser.parse_args()
print(opt)

# -------------------- Main pipeline --------------------
def main():
    # -------------------- 1. Dataset paths --------------------
    # universal path relative to current script for portability
    base_path = os.path.join(os.path.dirname(__file__), 'datasets')
    train_path = os.path.join(base_path, opt.dataset, 'train.txt')
    test_path  = os.path.join(base_path, opt.dataset, 'test.txt')

    # -------------------- 2. Load pickle train/test splits --------------------
    train_data = pickle.load(open(train_path, 'rb'))
    test_data  = pickle.load(open(test_path,  'rb'))

    # -------------------- 3. Dataset vocab size --------------------
    if opt.dataset == 'diginetica': n_node = 43097
    elif opt.dataset == 'Tmall': n_node = 40727
    elif opt.dataset == 'Nowplaying': n_node = 60416
    else: n_node = 309  # fallback

    # -------------------- 4. Wrap into Data objects --------------------
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data  = Data(test_data, shuffle=True, n_node=n_node)

    # -------------------- 5. Optional: truncate data for quick debugging --------------------
    if opt.cut_data:
        train_data.raw = train_data.raw[:opt.cut_data]
        train_data.targets = train_data.targets[:opt.cut_data]
        train_data.length = len(train_data.raw)

        test_data.raw = test_data.raw[:opt.cut_data]
        test_data.targets = test_data.targets[:opt.cut_data]
        test_data.length = len(test_data.raw)

    # -------------------- 6. Instantiate HCGATCL-SR model --------------------
    model = trans_to_cpu(
        HCGATCL_SR(
            adjacency=train_data.adjacency,
            n_node=n_node,
            lr=opt.lr,
            l2=opt.l2,
            beta=opt.beta,
            layers=int(opt.layer),
            emb_size=opt.embSize,
            batch_size=opt.batchSize,
            dataset=opt.dataset,
            hard_k=opt.hard_k,
            temperature=opt.temperature
        )
    )

    # -------------------- 7. Prepare metrics tracking --------------------
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results[f'epoch{K}']  = [0, 0]   # best epoch for P@K and MRR@K
        best_results[f'metric{K}'] = [0.0, 0.0]  # best values for P@K, MRR@K

    # -------------------- 8. Training loop --------------------
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        # train + evaluate one epoch
        metrics, total_loss = train_test(model, train_data, test_data)

        # compute average metrics for this epoch
        for K in top_K:
            metrics[f'precision{K}'] = np.mean(metrics[f'precision{K}']) * 100
            metrics[f'mrr{K}']       = np.mean(metrics[f'mrr{K}'])       * 100

            # update best metrics if current epoch is better
            if best_results[f'metric{K}'][0] < metrics[f'precision{K}']:
                best_results[f'metric{K}'][0] = metrics[f'precision{K}']
                best_results[f'epoch{K}'][0]  = epoch
            if best_results[f'metric{K}'][1] < metrics[f'mrr{K}']:
                best_results[f'metric{K}'][1] = metrics[f'mrr{K}']
                best_results[f'epoch{K}'][1]  = epoch

        # print best values so far
        for K in top_K:
            print('train_loss:\t%.4f\tPrecision@%d: %.4f\tMRR%d: %.4f\tEpoch: %d, %d' % (
                total_loss, K,
                best_results[f'metric{K}'][0], K, best_results[f'metric{K}'][1],
                best_results[f'epoch{K}'][0], best_results[f'epoch{K}'][1]))


if __name__ == '__main__':
    main()
