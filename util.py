import numpy as np
from scipy.sparse import csr_matrix

"""
    根据会话数据生成稀疏矩阵的掩码。
    参数:
        all_sessions (list): 所有会话的列表，每个会话是一个序列。
        n_node (int): 节点数量。
    返回值：
        matrix (csr_matrix): 稀疏矩阵，表示会话数据的掩码。
"""
def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

"""
    将数据集划分为训练集和测试集。
    参数:
        train_set (tuple): 包含训练集特征和标签的元组。
        valid_portion (float): 测试集所占的比例。
    返回值:
        (train_set, valid_set) (tuple): 划分后的训练集和测试集。
"""
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

"""
    数据类，用于存储和处理原始数据。
    参数:
        data (tuple): 包含原始数据特征和标签的元组。
        shuffle (bool): 是否对数据重排列。
        n_node (int): 节点数量。
    属性:
        raw (ndarray): 原始数据特征。
        adjacency (coo_matrix): 数据的邻接矩阵。
        n_node (int): 节点数量。
        targets (ndarray): 数据的标签。
        length (int): 数据长度。
        shuffle (bool): 是否对数据重排列。
"""
class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0], dtype=object)
        H_T = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    """
        这个函数计算给定会话列表中每对会话之间的重叠度量。
        参数:
            sessions (list): 包含会话的列表。
        返回值:
            matrix (ndarray): 重叠度量矩阵。
            degree (ndarray): 度矩阵。
        处理步骤：
            （1）首先创建一个大小为 (len(sessions), len(sessions)) 的零矩阵。
            （2）然后，对于每对不同的会话，将会话转换为集合，并移除其中的零元素。
            （3）接下来，计算两个会话之间的重叠元素集合以及两个会话的并集。
            （4）最后，返回重叠度量矩阵和度矩阵作为结果。
    """
    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    """
       这段代码用于生成批量数据切片。
       参数:
           batch_size (int): 批量大小。
       返回值:
           slices (list): 数据切片列表。
       处理步骤：
          （1）如果设置了shuffle参数为True，它将对数据进行随机打乱。
          （2）然后，它计算批量的数量，并确定最后一个批次的索引范围。
          （3）最后，它将数据切片分成相应的批次，并返回数据切片列表。
    """
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    """
       这段代码用于获取给定索引的数据切片。
       参数:
           index (int): 数据索引。
       返回值:
           targets (ndarray): 切片的标签。
           session_len (list): 切片中每个会话的长度列表。
           items (list): 切片中每个会话的填充项列表。
           reversed_sess_item (list): 切片中每个会话的反转填充项列表。
           mask (list): 切片中每个会话的掩码列表。
       处理步骤：
          （1）它首先获取指定索引的原始数据项，并计算每个会话的长度。
          （2）然后，它确定了切片中会话的最大节点数。
          （3）接下来，它为每个会话生成填充项、掩码和反转填充项，以使每个会话具有相同的长度。
          （4）最后，它返回切片的标签、会话长度列表、填充项列表、反转填充项列表和掩码列表。
    """
    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])

        return self.targets[index]-1, session_len,items, reversed_sess_item, mask
