import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from numba import jit
import heapq


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''
HyperConv类继承自Module的模型类，实现了超图卷积操作，对应论文公式(3.1)、(3.2)、(3.3)。
'''
class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size                                             # emb_size参数指定输入的嵌入维度。
        self.layers = layers                                                 # layers参数指定超图卷积的层数。
        self.dataset = dataset                                               # dataset参数指定数据集的名称。

    def forward(self, adjacency, embedding):                                 # forward方法定义了模型的前向传播过程，输入是邻接矩阵adjacency和嵌入矩阵 embedding。
        item_embeddings = embedding                                          # 将item_embeddings变量初始化为embedding。
        item_embedding_layer0 = item_embeddings                              # item_embedding_layer0是初始的item_embeddings。
        final = [item_embedding_layer0]                                      # final列表用于保存每一层的item_embeddings。
        for i in range(self.layers):                                         # 使用循环进行layers层的超图卷积操作。
            item_embeddings = torch.sparse.mm(trans_to_cpu(adjacency),      # 使用torch.sparse.mm将adjacency和item_embeddings进行稀疏矩阵乘法。
                                              item_embeddings)
            item_embeddings = F.tanh(item_embeddings)                        # 添加激活函数处理item_embeddings。
            final.append(item_embeddings)                                    # 将处理后的item_embeddings添加到final列表。
        #item_embeddings = np.sum(final, 0) / (self.layers + 1)               # 对所有层的item_embeddings求和并除以层数加一，得到最终的item_embeddings。
        item_embeddings = torch.stack(final, dim=0).mean(dim=0)
        return item_embeddings                                               # 返回最终的item_embeddings。

'''
GraphAttentionLayer类继承自Module的模型类，实现了图注意力操作，对应论文公式(3.7)、(3.8)。
'''
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features                                       # in_features参数指定输入特征的维度。
        self.out_features = out_features                                     # out_features参数指定输出特征的维度。
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))) # W是一个可训练参数，表示注意力机制中的权重矩阵，维度为(in_features, out_features)。
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))         # a是一个可训练参数，表示注意力机制中的参数矩阵，维度为(2*out_features, 1)。
        self.leakyrelu = nn.LeakyReLU(0.2)                                   # leakyrelu激活函数。
        self.reset_parameters()                                              # reset_parameters方法用于初始化权重矩阵W和参数矩阵a。


    def reset_parameters(self):                                              #reset_parameters方法用于初始化模型的参数。
        nn.init.xavier_uniform_(self.W.data, gain=1.414)                     #self.W.data和self.a.data表示参数张量的数据部分。
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adjacency):                                     # 前向传播的forward方法接收图的节点特征input和邻接矩阵adjacency。
        h = torch.matmul(input, self.W)                                      # 将输入特征input与权重矩阵W相乘得到h。
        N = h.size()[0]                                                      # 获取输入特征维度h的大小。

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)],# 构造输入a_input，将h在不同维度上进行重复并拼接，形成注意力机制的输入。
                             dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))         # 通过矩阵乘法和激活函数得到注意力系数e。

        zero_vec = -9e15 * torch.ones_like(e)                                # 创建一个与e大小相同的零向量zero_vec。
        attention = torch.where(adjacency > 0, e, zero_vec)                  # 根据邻接矩阵adjacency和e，将小于零的注意力系数设置为zero_vec，掩盖无关的邻居节点。

        attention = F.softmax(attention, dim=1)                              # 对注意力系数进行softmax 归一化，得到注意力权重attention。
        h_prime = torch.matmul(attention, h)                                 # 使用注意力权重对输入特征h进行加权求和。

        return h_prime


'''
GAT类继承自Module的模型类，实现了图注意力传播，对应论文公式(3.9)。
'''
class GAT(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(GAT, self).__init__()
        self.emb_size = emb_size                                             # emb_size参数指定嵌入特征的维度。
        self.batch_size = batch_size                                         # batch_size参数指定批量大小。
        self.layers = layers                                                 # layers参数指定图注意力层的数量。
        self.attention_layers = nn.ModuleList([                              # attention_layers用于存储多个图注意力层对象。
            GraphAttentionLayer(emb_size, emb_size) for _ in range(layers)
        ])

    def forward(self, item_embedding, D, A, session_item, session_len):        # 前向传播接收输入的item_embedding、度矩阵D、邻接矩阵A、会话项目session_item和会话长度session_len。
        #zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)              # 创建一个大小为 (1, emb_size) 的零向量 zeros。
        zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)                 # 将zeros与item_embedding在维度0上进行拼接，用于在起始位置添加一个零向量。
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding,                    # 使用torch.index_select根据session_item中的索引从item_embedding中选择对应的项目特征，将其添加到seq_h列表中。
                                            0, session_item[i]))
        seq_h1 = trans_to_cpu(torch.tensor(                                   # 转换为CUDA张量。
            np.array([item.cpu().detach().numpy() for item in seq_h])))
        session_emb_sggat = torch.div(torch.sum(seq_h1, 1), session_len)       # 计算会话的平均嵌入特征session_emb_sggat，对seq_h1进行求和后除以session_len。
        session = [session_emb_sggat]                                          # 创建一个session列表，初始时包含session_emb_sggat。
        DA = torch.mm(D, A).float()                                            # 计算DA，即特征矩阵D与邻接矩阵A的矩阵乘法结果，并转换为浮点型。
        for i in range(self.layers):
            session_emb_sggat = self.attention_layers[i](session_emb_sggat, DA) # 使用图注意力层处理嵌入特征和DA。
            session.append(session_emb_sggat)                                   # 将处理后的特征添加到会话列表。
        session1 = trans_to_cpu(torch.tensor(np.array([item.cpu().detach().numpy() for item in session])))
        session_emb_sggat = torch.sum(session1, 0)                              # 计算会话嵌入特征的总和。
        return session_emb_sggat                                                # 返回会话嵌入特征。


class SR_HCGAT(Module):

    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, emb_size=100, batch_size=100):
        super(SR_HCGAT, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset
        '''
        init方法：
            （1）初始化模型参数和超参数：根据输入的参数初始化模型的各个参数和超参数。
            （2）处理邻接矩阵：将给定的邻接矩阵数据转换为稀疏张量格式。
            （3）定义模型结构：创建模型的各个组件，包括节点嵌入层、位置嵌入层、超图卷积层、图注意力层等。
            （4）定义损失函数和优化器：设置模型的损失函数为交叉熵损失函数，并使用Adam优化器进行参数优化。
            （5）初始化模型参数：对模型的参数进行初始化操作，以确保模型在训练前处于良好的状态。
        '''
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = np.array([indices1, indices2])
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        #adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        adjacency = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset)
        self.LineGraph = GAT(self.layers, self.batch_size)
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.hyper_conv = HyperConv(layers, dataset, emb_size)
        self.line_conv = GAT(layers, batch_size, emb_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    '''
    用于实现软注意力机制，对应论文公式(3.4)、(3.5)、(3.6)。
    参数：
        item_embedding：形状为 (n_items, emb_size) 的项目嵌入矩阵。
        session_item：形状为 (batch_size, session_length) 的会话项目序列。
        session_len：形状为 (batch_size,) 的会话长度序列。
        reversed_sess_item：形状为 (batch_size, session_length) 的反转后的会话项目序列。
        mask：形状为 (batch_size, session_length) 的掩码矩阵。
    返回值：
        select：根据计算得到的注意力权重beta对序列嵌入seq_h进行加权求和后得到的向量。
    处理步骤：
        （1）将输入的项目嵌入向量与一个全零向量拼接，得到扩展后的项目嵌入向量。
        （2）构建一个张量，用于存储会话中每个项目的嵌入向量，将会话中的每个项目的嵌入向量存储在该张量中。
        （3）对会话中的项目嵌入向量进行求和并除以会话长度，得到整个会话的嵌入向量。
        （4）对位置嵌入进行处理，使其与项目嵌入向量的长度相同。
        （5）将位置嵌入和项目嵌入向量进行拼接，并通过线性层进行变换。
        （6）对变换后的结果应用激活函数，并与会话嵌入向量相加。
        （7）使用注意力机制对加权后的嵌入向量进行计算，返回加权后的嵌入向量，表示整个会话的表示。
    '''
    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        #zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        #seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        #zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        #seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        # pos_emb = self.pos_embedding.weight[:len]
        # pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    '''
        该代码段实现了一种基于自监督学习的损失函数，通过最大化正例得分和最小化负例得分的差异，用来训练模型。
        参数：
            sess_emb_hgnn：HGCN生成的会话嵌入的张量。
            sess_emb_gat：GAT生成的会话嵌入的张量。
        返回值：
            con_loss：表示在自监督学习期间计算的对比损失的标量张量。
        处理步骤：
            （1）正得分计算：使用两个输入张量计算正得分。
            （2）负得分计算：对其中一个输入张量进行行列乱序操作，然后与另一个输入张量计算负得分。
            （3）创建一个全为1的张量，与负得分张量形状相同。
            （4）对比损失计算：通过应用sigmoid函数和对数损失，计算正得分和负得分之间的对比损失。
            （5）输出：返回对比损失作为函数的输出。
    '''

#    def NT_Xent_loss(self, z1, z2, temperature=0.1):
#        """Contrastive loss based on NT-Xent (SimCLR-style)."""
#        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
#        logits = torch.mm(z1, z2.t()) / temperature
#        labels = torch.arange(logits.size(0)).to(logits.device)
#        loss = F.cross_entropy(logits, labels)
#        return loss

    def info_nce_loss(self, z1, z2, temperature=0.1):
        """
        InfoNCE: считаем logits z1 @ z2.T / t и cross-entropy по диагонали.
        z1, z2 – (B, D), уже нормализованы.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.mm(z1, z2.t()) / temperature          # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss   = F.cross_entropy(logits, labels)             # softmax по строкам
        return loss

    def dropout_noise(self, emb, drop_rate=0.1):
        """Apply dropout noise for augmentation."""
        mask = torch.rand_like(emb) > drop_rate
        return emb * mask.float()

    def SSL(self, sess_emb_hgnn, sess_emb_gat):
        """Self-supervised learning with NT-Xent loss and dropout augmentation."""
        z1 = self.dropout_noise(sess_emb_hgnn)
        z2 = self.dropout_noise(sess_emb_gat)
#        con_loss = self.NT_Xent_loss(z1, z2)
        con_loss = self.info_nce_loss(z1, z2, temperature=0.1)   # температуру можно вынести в self.t
        return con_loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        if self.dataset == 'Tmall':
            sess_emb_hgnn = self.generate_sess_emb_npos(item_embeddings_hg, session_item, session_len,
                                                        reversed_sess_item, mask)
        else:
            sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item,
                                                   mask)
        session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg)
        return item_embeddings_hg, sess_emb_hgnn, self.beta * con_loss

'''
    该代码段从给定的候选项列表中找到前K个最大的值，并返回它们的索引。
    参数：
        K：要找到的最大值的数量。
        candidates：包含候选项分数的列表。
    返回值：
        ids：返回一个列表，即找到的前K个最大值的索引。
    处理步骤：
        （1）创建一个空列表用于存储最大的K个候选项，每个候选项由其分数和索引组成。
        （2）遍历candidates列表中前K个元素的索引和值，将它们作为元组添加到candidates列表中。
        （3）将n_candidates列表转换为堆结构，以便能够高效地找到最小的元素。
        （4）遍历candidates列表中第K个元素及之后的元素的索引和值：如果当前值大于堆中的最小值，则将当前值替换为堆中的最小值，并更新对应的索引。
        （5）降序排列后，返回ids列表，即找到的前K个最大值的索引。
'''
@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids  # , k_largest_scores

'''
    该代码在给定模型、数据索引和数据对象的情况下，执行模型的前向传播计算，并返回计算结果。
    参数：
        model：模型对象
        i：数据索引
        data：数据对象
    返回值：
        tar：目标张量
        scores：分数张量
        con_loss：损失张量
    （1）获取输入数据的切片。
    （2）获取session_item的重叠信息。
    （3）对变量分别应用trans_to_cpu函数转换为CUDA张量。
    （4）将输入数据传递给模型，计算sess_emb_hgnn和item_emb_hg的矩阵乘法，得到scores。
    （5）返回tar,scores,con_loss 作为输出结果。
'''
def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cpu(torch.Tensor(session_item).long())
    session_len = trans_to_cpu(torch.Tensor(session_len).long())
    A_hat = trans_to_cpu(torch.Tensor(A_hat))
    D_hat = trans_to_cpu(torch.Tensor(D_hat))
    tar = trans_to_cpu(torch.Tensor(tar).long())
    mask = trans_to_cpu(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cpu(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss

'''
    该代码的功能是训练和测试模型。在训练阶段，它通过迭代训练数据的批次，计算模型的预测得分和损失，并进行反向传播和参数更新，同时累计计算总损失。
    参数：
        model：模型对象
        train_data：训练数据对象
        test_data：测试数据对象
    返回值：
        metrics：包含评估指标和总损失的字典
        total_loss：训练过程中的总损失值 
    训练阶段：（1）通过迭代训练数据批次，调用forward函数计算模型预测得分和损失。
            （2）进行反向传播和参数更新，同时累计计算总损失。
    测试阶段：（1）使用模型对测试数据进行预测，并计算评估指标（如命中率和倒数排名）来衡量模型性能。
            （2）返回计算得到的评估指标和总损失作为结果。
'''
from tqdm import tqdm

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i in tqdm(slices, desc="Training"):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['precision%d' % K] = []
        metrics['mrr%d' % K] = []

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)

    for i in tqdm(slices, desc="Testing"):
        tar, scores, con_loss = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['precision%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))

    return metrics, total_loss


