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
from tqdm import tqdm

# -------------------- CPU helper --------------------
def trans_to_cpu(variable):
    """Transfer tensor to CPU if using GPU; otherwise, no-op."""
    if torch.cuda.is_available():
        return variable.cpu()
    return variable

# -------------------- 1. Hypergraph Convolution (item-level) --------------------
class HyperConv(nn.Module):
    """
    Propagates item embeddings over hypergraph adjacency with tanh activation.
    Returns mean embedding across all layers.
    """
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        final = [item_embeddings]

        for _ in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cpu(adjacency), item_embeddings)
            item_embeddings = torch.tanh(item_embeddings)
            final.append(item_embeddings)

        return torch.stack(final, dim=0).mean(dim=0)

# -------------------- 2. Single-head Graph Attention Layer --------------------
class GraphAttentionLayer(nn.Module):
    """
    Computes session-level attention for a single GAT layer.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2*out_features,1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size(0)
        # pairwise attention computation
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.mm(attention, h)
        return h_prime
# -------------------- 3. Multi-layer GAT --------------------
class GAT(nn.Module):
    """
    Multi-layer session-level attention using stacked GraphAttentionLayer.
    """
    def __init__(self, layers, batch_size, emb_size=100):
        super().__init__()
        self.layers = layers
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.attention_layers = nn.ModuleList([GraphAttentionLayer(emb_size, emb_size) for _ in range(layers)])

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding],0)
        # initial mean pooling session representation
        seq_h = [torch.index_select(item_embedding,0, s) for s in session_item]
        session_emb = torch.stack(seq_h)
        session_emb = torch.div(torch.sum(session_emb,1), session_len)
        # GAT propagation
        DA = torch.mm(D,A).float()
        session_list = [session_emb]
        for i in range(self.layers):
            session_emb = self.attention_layers[i](session_emb, DA)
            session_list.append(session_emb)
        return torch.stack(session_list,dim=0).mean(dim=0)

# -------------------- 4. HCGATCL-SR complete model --------------------
class HCGATCL_SR(nn.Module):
    """
    Combines Hypergraph item-level and session-level GAT representations.
    Includes soft attention, fusion layer, and hard negative contrastive learning.
    """
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset,
                 emb_size=100, batch_size=100, hard_k=10, temperature=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset
        self.hard_k = hard_k
        self.temperature = temperature

        # sparse adjacency matrix for hypergraph
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset=='Nowplaying':
            keep = values>=0.05
            values=values[keep]; indices=indices[:,keep]
        i = torch.LongTensor(indices); v = torch.FloatTensor(values)
        self.adjacency = torch.sparse_coo_tensor(i,v,torch.Size(adjacency.shape))

        # learnable embeddings
        self.embedding = nn.Embedding(n_node, emb_size)
        self.pos_embedding = nn.Embedding(200, emb_size)

        self.HyperGraph = HyperConv(layers, dataset, emb_size)
        self.SessionGraph = GAT(layers, batch_size, emb_size)

        self.w_1 = nn.Linear(2*emb_size, emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(emb_size,1))
        self.glu1 = nn.Linear(emb_size, emb_size)
        self.glu2 = nn.Linear(emb_size, emb_size, bias=False)

        # fusion layer for item+session views
        self.fusion_h = nn.Linear(emb_size, emb_size, bias=False)
        self.fusion_g = nn.Linear(emb_size, emb_size, bias=False)

        self.W_out = nn.Linear(emb_size, emb_size, bias=False)

        # loss and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)
        self.init_parameters()

    # initialize all weights uniformly
    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    # session encoder with positional attention
    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.zeros(1,self.emb_size)
        item_embedding = torch.cat([zeros,item_embedding],0)
        seq_h = torch.zeros(self.batch_size,reversed_sess_item.shape[1],self.emb_size)
        for i in torch.arange(session_item.shape[0]): seq_h[i]=item_embedding[reversed_sess_item[i].long()]
        hs = torch.div(torch.sum(seq_h,1),session_len)
        # positional attention
        mask = mask.float().unsqueeze(-1)
        length = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:length].unsqueeze(0).repeat(self.batch_size,1,1)
        hs = hs.unsqueeze(-2).repeat(1,length,1)
        nh = torch.tanh(self.w_1(torch.cat([pos_emb,seq_h],-1)))
        nh = torch.sigmoid(self.glu1(nh)+self.glu2(hs))
        beta = torch.matmul(nh,self.w_2)*mask
        select = torch.sum(beta*seq_h,1)
        return select

    # simplified encoder without positional embedding
    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.zeros(1,self.emb_size)
        item_embedding = torch.cat([zeros,item_embedding],0)
        seq_h = torch.zeros(self.batch_size,reversed_sess_item.shape[1],self.emb_size)
        for i in torch.arange(session_item.shape[0]): seq_h[i]=item_embedding[reversed_sess_item[i].long()]
        hs = torch.div(torch.sum(seq_h,1),session_len)
        mask = mask.float().unsqueeze(-1)
        hs = hs.unsqueeze(-2).repeat(1,seq_h.shape[1],1)
        nh = seq_h
        nh = torch.tanh(nh); nh = torch.sigmoid(self.glu1(nh)+self.glu2(hs))
        beta = torch.matmul(nh,self.w_2)*mask
        select = torch.sum(beta*seq_h,1)
        return select

    # contrastive loss with hard negative mining
    def HNM_InfoNCE_loss(self, z1, z2):
        """
        Hard Negative Mining InfoNCE loss.
        Positive pair: z1[i] and z2[i].
        Hard negatives: top-K most similar z2[j], where j != i.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        temperature = self.temperature

        # similarity matrix: (B, B)
        sim_matrix = torch.mm(z1, z2.t()) / temperature

        # positive scores: diagonal elements
        pos_scores = sim_matrix.diag().unsqueeze(1)  # (B, 1)

        # mask diagonal so that positive pairs are not selected as negatives
        neg_sim = sim_matrix.clone()
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        neg_sim = neg_sim.masked_fill(diag_mask, float('-inf'))

        # choose top-K hard negatives for each anchor
        k = min(self.hard_k, batch_size - 1)
        hard_neg_indices = torch.topk(neg_sim.detach(), k=k, dim=1).indices  # (B, K)

        # gather hard negative scores from original sim_matrix
        hard_neg_scores = torch.gather(sim_matrix, dim=1, index=hard_neg_indices)  # (B, K)

        # logits contain one positive + K hard negatives
        logits = torch.cat([pos_scores, hard_neg_scores], dim=1)  # (B, 1+K)

        # positive sample is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def dropout_noise(self, emb, drop_rate=0.1):
        mask = torch.rand_like(emb)>drop_rate
        return emb*mask.float()

    def SSL(self, sess_emb_hgnn, sess_emb_gat):
        z1 = self.dropout_noise(sess_emb_hgnn)
        z2 = self.dropout_noise(sess_emb_gat)

        loss_1 = self.HNM_InfoNCE_loss(z1, z2)  # hypergraph view as anchor
        loss_2 = self.HNM_InfoNCE_loss(z2, z1)  # GAT view as anchor

        return 0.5 * (loss_1 + loss_2)
    
    def fuse_session_representations(self,sess_emb_hgnn,sess_emb_gat):
        """
        Fuse item-level and session-level session representations.

        sess_emb_hgnn: item-level session representation from the HGCN branch
        sess_emb_gat : session-level session representation from the GAT branch

        return:
            fused session representation used for next-item prediction
        """
        sess_emb_fused = self.fusion_h(sess_emb_hgnn)+self.fusion_g(sess_emb_gat)
        return torch.tanh(sess_emb_fused)

    # full forward pass
    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask):
        # hypergraph view
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)

        # item-level session representation from the hypergraph branch
        if self.dataset == 'Tmall':
            sess_emb_hgnn = self.generate_sess_emb_npos(
                item_embeddings_hg, session_item, session_len,
                reversed_sess_item, mask)
        else:
            sess_emb_hgnn = self.generate_sess_emb(
                item_embeddings_hg, session_item, session_len,
                reversed_sess_item, mask)

        # session-level session representation from the GAT branch
        session_emb_lg = self.SessionGraph(
            self.embedding.weight, D, A, session_item, session_len)

        # contrastive learning loss between the two views
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg)

        # fused session representation used for prediction
        sess_emb_fused = self.fuse_session_representations(sess_emb_hgnn, session_emb_lg)

        return item_embeddings_hg, sess_emb_fused, self.beta * con_loss
    
# -------------------- 5. CPU-efficient top-K finder --------------------
@jit(nopython=True)
def find_k_largest(K, candidates):
    """
    Returns indices of top-K largest values in descending order.
    """
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)

    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))

    n_candidates.sort(key=lambda d: d[0], reverse=True)
    return [item[1] for item in n_candidates]

# -------------------- 6.  Mini-batch forward wrapper --------------------
def forward(model, i, data):
    """
    Gets one mini-batch, feeds it to the model, returns predictions & loss.
    """
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    A_hat, D_hat = data.get_overlap(session_item)

    # move everything to CPU (possible change to CUDA)
    session_item = trans_to_cpu(torch.tensor(session_item, dtype=torch.long))
    session_len  = trans_to_cpu(torch.tensor(session_len,  dtype=torch.long))
    A_hat        = trans_to_cpu(torch.tensor(A_hat, dtype=torch.float32))
    D_hat        = trans_to_cpu(torch.tensor(D_hat, dtype=torch.float32))
    tar          = trans_to_cpu(torch.tensor(tar,   dtype=torch.long))
    mask         = trans_to_cpu(torch.tensor(mask,  dtype=torch.long))
    reversed_sess_item = trans_to_cpu(torch.tensor(reversed_sess_item, dtype=torch.long))

    item_emb_hg, sess_emb_fused, con_loss = model(
        session_item, session_len, D_hat, A_hat, reversed_sess_item, mask)

    scores = torch.mm(model.W_out(sess_emb_fused), item_emb_hg.t())
    return tar, scores, con_loss

# -------------------- 7.  Training & evaluation loop --------------------
def train_test(model, train_data, test_data):
    """
    One full epoch: train on `train_data`, evaluate on `test_data`.
    Returns metrics dict and average training loss.
    """
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    # ---------- training ----------
    for i in tqdm(slices, desc="Training"):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets) + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

    print('\tLoss:\t%.3f' % total_loss)

    # ---------- evaluation ----------
    top_K = [5, 10, 20]
    metrics = {f'precision{k}': [] for k in top_K}
    metrics.update({f'mrr{k}': [] for k in top_K})

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    slices = test_data.generate_batch(model.batch_size)

    with torch.no_grad():
        for i in tqdm(slices, desc="Testing"):
            tar, scores, con_loss = forward(model, i, test_data)
            scores = trans_to_cpu(scores).numpy()

            # top-20 indices for each session in the batch
            index = [find_k_largest(20, scores[idd]) for idd in range(model.batch_size)]
            index = np.array(index)
            tar = trans_to_cpu(tar).numpy()

            for K in top_K:
                for pred, target in zip(index[:, :K], tar):
                    hit = np.isin(target, pred)
                    metrics[f'precision{K}'].append(float(hit))
                    idxs = np.where(pred == target)[0]
                    metrics[f'mrr{K}'].append(0.0 if len(idxs) == 0 else 1.0 / (idxs[0] + 1.0))

    return metrics, total_loss
