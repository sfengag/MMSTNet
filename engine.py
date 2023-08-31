import torch.optim as optim
from model import *
from Util_general import *


class trainer0():
    def __init__(self, device, num_nodes, num_regions, num_graphs, in_dim, in_dim_cluster, out_dim, channels_0, channels_1,
                 channels_2, support_list, transmit, mf_embedding_size, num_time_steps, Kt, K_ta, order_t, K_sa, lrate,
                 wdecay, decay, mean_val, std):

        self.model = HST(num_nodes, num_regions, num_graphs, in_dim, in_dim_cluster, out_dim, channels_0, channels_1,
                 channels_2, support_list, transmit, mf_embedding_size, num_time_steps, Kt, K_ta, order_t, K_sa)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = masked_mae
        self.clip = None
        self.support_list = support_list
        self.num_nodes = num_nodes
        self.num_regions = num_regions
        self.num_graphs = num_graphs
        self.mean = torch.tensor(mean_val).to(device)
        self.std = torch.tensor(std).to(device)


    def train(self, input, input_cluster, real, tem_embedding, spa_embedding):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(input, input_cluster, tem_embedding, spa_embedding, self.mean, self.std)
        pred = pred.transpose(1, 3)
        loss = self.loss(pred, real, 0.)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        MAE = masked_mae(pred,real, 0.).item()
        MAPE = masked_mape(pred,real, 0.).item()
        RMSE = masked_rmse(pred,real, 0.).item()

        return loss.item(), MAE, MAPE, RMSE

    def eval(self, input, input_cluster, real, tem_embedding, spa_embedding):
        self.model.eval()
        pred = self.model(input, input_cluster, tem_embedding, spa_embedding, self.mean, self.std)
        pred = pred.transpose(1, 3)
        loss = self.loss(pred, real, 0.)
        MAE = masked_mae(pred,real, 0.).item()
        MAPE = masked_mape(pred,real, 0.).item()
        RMSE = masked_rmse(pred,real, 0.).item()

        return loss.item(), MAE, MAPE, RMSE

