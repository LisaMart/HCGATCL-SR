import numpy as np
from scipy.sparse import csr_matrix

# -------------------- 1. Build transposed incidence matrix H^T --------------------
def data_masks(all_sessions, n_node):
    """
    Build the binary transposed incidence matrix H^T for the item-session hypergraph.
    Rows correspond to sessions (hyperedges), columns correspond to items.
    Entry H_T[j, i] = 1 if item v_i appears in session/hyperedge e_j, otherwise 0.
    This corresponds to the hypergraph construction before Eq. (4.1) in Chapter 4.
    """
    indptr, indices, data = [0], [], []
    for session in all_sessions:
        uniq = np.unique(session)
        indices.extend(uniq - 1)  # 0-based
        data.extend([1] * len(uniq))
        indptr.append(indptr[-1] + len(uniq))
    return csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node), dtype=np.float32)

# -------------------- 2. Random train/validation split --------------------
def split_validation(train_set, valid_portion):
    """
    Split a dataset into training and validation sets.
    Randomly shuffles indices before splitting.
    Returns two tuples: (train_x, train_y), (valid_x, valid_y)
    """
    x, y = train_set
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    n_train = int(len(x) * (1 - valid_portion))
    return (x[idx[:n_train]], y[idx[:n_train]]), (x[idx[n_train:]], y[idx[n_train:]])

# -------------------- 3. Data container --------------------
class Data:
    """
    Holds a split (train/valid/test) and constructs:
    - item-item adjacency matrix DH·BH^T for hypergraph convolution
    - session overlap matrices for GAT
    Provides padded mini-batch generator.
    """
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.array(data[0], dtype=object)
        self.targets = np.array(data[1], dtype=np.int64)
        self.n_node = n_node
        self.length = len(self.raw)
        self.shuffle = shuffle

        # build adjacency matrix
        H_T = data_masks(self.raw, n_node)
        row_sum = np.asarray(H_T.sum(axis=1)).ravel(); row_sum[row_sum == 0] = 1
        BH_T = H_T.multiply(1.0 / row_sum.reshape(-1,1))
        H = H_T.T
        col_sum = np.asarray(H.sum(axis=1)).ravel(); col_sum[col_sum == 0] = 1
        DH = H.multiply(1.0 / col_sum.reshape(-1,1))
        self.adjacency = (DH @ BH_T).tocoo()

    # compute similarity between sessions
    # Similarity score is corresponding to Eq. (4.1) in the dissertation
    def get_overlap(self, sessions):
        B = len(sessions)
        matrix = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            s_i = set(sessions[i]) - {0}
            for j in range(i+1, B):
                s_j = set(sessions[j]) - {0}
                union = len(s_i | s_j)
                matrix[i][j] = matrix[j][i] = len(s_i & s_j)/union if union>0 else 0.0
        matrix += np.diag([1.0]*B)
        degree = np.diag(1.0 / np.sum(matrix, axis=1))
        return matrix, degree

    # generate mini-batch indices
    def generate_batch(self, batch_size):
        if self.shuffle:
            idx = np.arange(self.length)
            np.random.shuffle(idx)
            self.raw = self.raw[idx]
            self.targets = self.targets[idx]
        n_batch = (self.length + batch_size - 1) // batch_size
        slices = np.split(np.arange(n_batch*batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    # get padded batch tensors
    def get_slice(self, index):
        inp = self.raw[index]
        session_len, items, reversed_sess_item, mask = [], [], [], []
        max_len = max(len(s[np.nonzero(s)]) for s in inp)
        for s in inp:
            l = len(np.nonzero(s)[0])
            session_len.append([l])
            items.append(s + [0]*(max_len-l))
            reversed_sess_item.append(list(reversed(s)) + [0]*(max_len-l))
            mask.append([1]*l + [0]*(max_len-l))
        return self.targets[index]-1, session_len, items, reversed_sess_item, mask【101†source】
