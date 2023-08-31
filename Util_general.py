import pickle
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

class DataLoader_with_cluster(object):
    def __init__(self, x, y, x_c, tem_embedding, batch_size, pad_last_example = True):
        self.batch_size = batch_size
        self.current_id = 0

        if pad_last_example:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            x_padding = np.repeat(x[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            x_c_padding = np.repeat(x_c[-1:], num_padding, axis=0)
            tem_em_padding = np.repeat(tem_embedding[-1:], num_padding, axis=0)

            x = np.concatenate([x, x_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
            x_c = np.concatenate([x_c, x_c_padding], axis=0)
            tem_embedding = np.concatenate([tem_embedding, tem_em_padding], axis=0)

        self.num_sample = len(x)
        self.num_batch = int(self.num_sample // self.batch_size)
        self.x = x
        self.y = y
        self.x_c = x_c
        self.tem_embedding = tem_embedding

    def shuffle(self):
        permutation = np.random.permutation(self.num_sample)
        self.x = self.x[permutation]
        self.y = self.y[permutation]
        self.x_c = self.x_c[permutation]
        self.tem_embedding = self.tem_embedding[permutation]

    def get_iterator(self):
        self.current_id = 0

        def _wrapper():
            while self.current_id < self.num_batch:
                start_id = self.batch_size * self.current_id
                end_id = min(self.num_sample, self.batch_size * (self.current_id + 1))
                x_batch = self.x[start_id: end_id, ...]
                y_batch = self.y[start_id: end_id, ...]
                x_c_batch = self.x_c[start_id: end_id, ...]
                tem_embedding_batch = self.tem_embedding[start_id: end_id, ...]
                yield (x_batch, y_batch, x_c_batch, tem_embedding_batch)
                self.current_id += 1

        return _wrapper()


def load_dataset_with_cluster(data_path, batch_size, cluster_type = 'fusion'):
    data = {}

    for f in ['train', 'valid', 'test']:
        original_data = np.load(data_path + '\\' + f  + '_z.npz')
        data['x_' + f] = original_data['x']
        data['y_' + f] = original_data['y']
        data['tem_em_' + f] = original_data['tem_embedding']
        if cluster_type == 'fusion':
            data['x_c_' + f] = original_data['x_c']
        elif cluster_type == 'distance':
            data['x_c_' + f] = original_data['x_c_dis']
        elif cluster_type == 'speed':
            data['x_c_' + f] = original_data['x_c_speed']

    data['train_loader'] = DataLoader_with_cluster(data['x_train'], data['y_train'], data['x_c_train'],
                                                      data['tem_em_train'], batch_size)
    data['valid_loader'] = DataLoader_with_cluster(data['x_valid'], data['y_valid'], data['x_c_valid'],
                                                   data['tem_em_valid'], batch_size)
    data['test_loader'] = DataLoader_with_cluster(data['x_test'], data['y_test'], data['x_c_test'],
                                                   data['tem_em_test'], batch_size)

    return data

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def generate_support(adj, type):
    support = []
    if type == 'double_transition':
        support = [asym_adj(adj), asym_adj(np.transpose(adj))]
    elif type == "scalap":
        support = [calculate_scaled_laplacian(adj)]
    else:
        error = 0
        assert error, 'not defined adj type'

    return support

def generate_support_list(data_path, graph_type, processing_type):
    adj_list = []
    support_list = []
    if graph_type == 'distance_and_speed_similarity':
        adj_dis = pickle.load(open(data_path + '\\' + 'adj_dis.pickle', 'rb'))
        adj_similarity = pickle.load(open(data_path + '\\' + 'adj_similarity.pickle', 'rb'))
        adj_list = [adj_dis, adj_similarity]
    elif graph_type == 'distance':
        adj = pickle.load(open(data_path + '\\' + 'adj_dis.pickle', 'rb'))
        adj_list = [adj]
    elif graph_type == 'speed':
        adj = pickle.load(open(data_path + '\\' + 'adj_similarity.pickle', 'rb'))
        adj_list = [adj]
    elif graph_type == 'cluster':
        adj = pickle.load(open(data_path + '\\' + 'adj_cluster.pickle', 'rb'))
        adj_list = [adj]
    elif graph_type == 'distance_cluster':
        adj = pickle.load(open(data_path + '\\' + 'adj_dis_cluster.pickle', 'rb'))
        adj_list = [adj]
    elif graph_type == 'speed_cluster':
        adj = pickle.load(open(data_path + '\\' + 'adj_similarity_cluster.pickle', 'rb'))
        adj_list = [adj]
    for adj in adj_list:
        support_list.append( generate_support(adj, processing_type) )

    return support_list

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.).item()
    mape = masked_mape(pred,real,0.).item()
    rmse = masked_rmse(pred,real,0.).item()
    return mae,mape,rmse


