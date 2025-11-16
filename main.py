import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os
from tqdm import tqdm


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="/root/autodl-tmp/logs/N_N")

'''超参数设定'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/Nowplaying/Tmall')
parser.add_argument('--epoch', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--cut_data', type=int, default=1000, help='use only first N sessions for quick test') # None для быстрой проверки

opt = parser.parse_args()
print(opt)

def main():
    # 加载训练数据与测试数据
    train_data = pickle.load(open('C:/Users/Chaingun/Desktop/SR-HCGAT_main/data/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('C:/Users/Chaingun/Desktop/SR-HCGAT_main/data/' + opt.dataset + '/test.txt', 'rb'))

    # 根据数据集选择节点数量
    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        n_node = 309

    # 创建训练数据和测试数据对象
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)

    # --- резкое уменьшение объёма ---
    if opt.cut_data:
        train_data.raw = train_data.raw[:opt.cut_data]
        train_data.targets = train_data.targets[:opt.cut_data]
        train_data.length = len(train_data.raw)

        test_data.raw = test_data.raw[:opt.cut_data]
        test_data.targets = test_data.targets[:opt.cut_data]
        test_data.length = len(test_data.raw)

    # 创建SR_HCGAT模型并转移到GPU上
    model = trans_to_cpu(SR_HCGAT(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,
                               emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    top_K = [5, 10, 20]
    # 存储最佳结果的字典
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]    # 最佳结果的epoch
        best_results['metric%d' % K] = [0, 0]   # 最佳结果的指标值

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        # 在训练集和测试集上进行训练和测试，并计算指标和总损失
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['precision%d' % K] = np.mean(metrics['precision%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100

            # 更新最佳结果的指标值和对应的epoch
            if best_results['metric%d' % K][0] < metrics['precision%d' % K]:
                best_results['metric%d' % K][0] = metrics['precision%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch

        # ✅ Чистый вывод без print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tPrecision@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

if __name__ == '__main__':
    main()
