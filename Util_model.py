import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

class Transmit(nn.Module):
    def __init__(self, c_in, c_in_1, embedding_size, num_time_steps, transmit, num_nodes, num_regions):
        super(Transmit, self).__init__()
        self.transmit = transmit
        self.num_regions = num_regions
        self.num_non_zero = torch.sum(transmit, dim=0)
        self.c_in = c_in
        self.c_in_1 = c_in_1
        self.conv1 = Conv2d(num_time_steps, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_c_1 = Conv2d(num_time_steps, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.w_r_em = nn.Parameter(torch.rand(num_regions, embedding_size), requires_grad=True)
        self.w_p_em = nn.Parameter(torch.rand(embedding_size, c_in*c_in_1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w_r_em)
        torch.nn.init.xavier_uniform_(self.w_p_em)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_regions), requires_grad=True)

    def forward(self, x, x_c):
        f1 = x.permute(0, 3, 2, 1) # b,c,n,t -> b,t,n,c
        f1 = self.conv1(f1).squeeze(1) # b,n,c
        f2 = x_c.permute(0, 3, 2, 1)  # b,c,n,t->b,t,n,c
        f2 = self.conv_c_1(f2).squeeze(1) # b,n,c

        # transform region feature with region specific transformation
        w = torch.mm(self.w_r_em, self.w_p_em)
        w = w.view((self.num_regions, self.c_in, self.c_in_1)) # n, c, c1
        f2 = torch.einsum('bnc,ncd->bnd', f2, w) # b,n,c1

        # attention and revise transmit matrix
        f2 = f2.permute(0, 2, 1) # b,c1,n
        logits = torch.sigmoid(torch.matmul(f1, f2) + self.b) # b,n,n
        logits = logits * self.transmit
        avg = torch.div(torch.sum(logits, dim=1, keepdim=True), self.num_non_zero)
        logits = logits - avg
        logits = torch.sigmoid(logits)
        coefs = logits * self.transmit

        return coefs

class TATT(nn.Module):
    def __init__(self, c_in, c_out, K, d, mask = True):
        super(TATT, self).__init__()
        # K: number of attention heads
        # d: dimension of attention outputs
        self.d = d
        self.K = K
        self.mask = mask
        self.conv_q = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_k = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_v = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_o = Conv2d(c_out, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.small_value = torch.tensor(-2**15+1.).cuda(0)


    def forward(self, x, tem_embedding):
        batch_size_ = x.shape[0]
        x = torch.cat((x, tem_embedding[:,:,:,-x.shape[3]:]), dim=1)
        query = self.conv_q(x)
        key = self.conv_k(x)
        value = self.conv_v(x)
        query = torch.cat(torch.split(query, self.d, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=1), dim=0)
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 3, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        if self.mask:
            batch_size = x.shape[0]
            num_step = x.shape[3]
            num_vertex = x.shape[2]
            mask = torch.ones(num_step, num_step).cuda(0)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, self.small_value)
        # softmax
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 3, 1, 2)
        x = torch.cat(torch.split(x, batch_size_, dim=0), dim=1)
        x = F.relu( self.conv_o(x) )
        del query, key, value, attention

        return x

class SATT(nn.Module):
    def __init__(self, c_in, c_out, K, d):
        super(SATT, self).__init__()
        # K: number of attention heads
        # d: dimension of attention outputs
        self.d = d
        self.K = K
        self.conv_q = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_k = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_v = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv_o = Conv2d(c_out, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, spa_embedding):
        batch_size_ = x.shape[0]
        x = torch.cat((x, spa_embedding[:,:,:,-x.shape[3]:]), dim=1)
        query = self.conv_q(x)
        key = self.conv_k(x)
        value = self.conv_v(x)
        query = torch.cat(torch.split(query, self.d, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=1), dim=0)
        query = query.permute(0, 3, 2, 1)
        key = key.permute(0, 3, 1, 2)
        value = value.permute(0, 3, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 3, 2, 1)
        x = torch.cat(torch.split(x, batch_size_, dim=0), dim=1)
        x = F.relu(self.conv_o(x))
        del query, key, value, attention

        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class MGC(nn.Module):
    def __init__(self, c_in, c_out, num_graphs = 2, support_len=2, order=3, dropout=0.):
        super(MGC, self).__init__()
        self.nconv = nconv()
        c_in = ((order-1) * support_len + 1) * c_in
        self.num_graphs = num_graphs
        self.conv_list= nn.ModuleList([Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
                                       for i in range(num_graphs)])
        self.order = order
        self.dropout = dropout

    def forward(self, x, support_list):
        out_list = [[x] for i in range(self.num_graphs)]
        h_list = []
        for i, support in enumerate(support_list):
            for A in support:
                x1 = self.nconv(x, A)
                out_list[i].append(x1)
                for k in range(2, self.order):
                    x2 = self.nconv(x1, A)
                    out_list[i].append(x2)
                    x1 = x2
        for i in range(self.num_graphs):
            h = torch.cat(out_list[i],dim=1)
            h = F.relu(self.conv_list[i](h))
            if self.dropout >0.:
                h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)

        return h_list

class micro_fusion(nn.Module):
    def __init__(self, c_in, c_out, num_graphs = 2):
        super(micro_fusion, self).__init__()
        self.conv_adp = Conv2d(c_in*num_graphs, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.conv_0 = Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=False)
        self.num_graphs = num_graphs

    def forward(self, x_list):
        if self.num_graphs > 1:
            # adpative context embedding
            u_adp = torch.cat(x_list, dim=1)
            u_adp = self.conv_adp(u_adp)

            h0 = torch.cat(x_list, dim=0)
            h0 = self.conv_0(h0)
            d = int(h0.shape[0] // self.num_graphs)
            h_list = list(torch.split(h0, d, dim=0))
            for i in range(self.num_graphs):
                h_list[i] = torch.unsqueeze(h_list[i],dim=-1)
            h = torch.cat(h_list, dim=-1)

            # attention
            attention = torch.einsum('bcntg, bcnt->bntg', h, u_adp)
            attention = F.softmax(attention, dim=-1)
            attention = torch.unsqueeze(attention, dim=1)
            x = torch.sum( torch.mul(attention, h), dim=-1)

        else:
            x = x_list[0]

        return x

class micro_block(nn.Module):
    def __init__(self, c_in, c_out, dim_tem_em, Kt, K_ta, d_ta, mask, num_graphs, support_len=2, order=2):
        super(micro_block, self).__init__()
        self.TC = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 0), stride=(1, 1), bias=True, dilation=(1, 2))
        self.TAT = TATT(c_out+dim_tem_em, c_out, K_ta, d_ta, mask)
        self.GC = MGC(c_out, c_out, num_graphs, support_len, order)
        self.MF = micro_fusion(c_out, c_out, num_graphs)


    def forward(self, x, support_list, tem_embedding):
        # TCN
        x = self.TC(x)
        x = F.relu(x)

        # temporal attention with temporal embedding
        x = self.TAT(x, tem_embedding)

        # multi graph GCN
        x_list = self.GC(x, support_list)

        # micro learning fusion
        x = self.MF(x_list)

        return x

class macro_block(nn.Module):
    def __init__(self, c_in, c_out, dim_tem_em, dim_spa_em, Kt, K_sa, d_sa, K_ta, d_ta, mask):
        super(macro_block, self).__init__()
        self.TC = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 0), stride=(1, 1), bias=True, dilation=(1, 2))
        self.TAT = TATT(c_out+dim_tem_em, c_out, K_ta, d_ta, mask)
        self.SAT = SATT(c_out+dim_spa_em, c_out, K_sa, d_sa)

    def forward(self, x, tem_embedding, spa_embedding):
        # TCN
        x = self.TC(x)
        x = F.relu(x)

        # temporal attention with temporal embedding
        x = self.TAT(x, tem_embedding)

        # spatial attention
        x = self.SAT(x, spa_embedding)

        return x
