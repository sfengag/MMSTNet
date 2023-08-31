import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm
from Util_model import micro_block, macro_block, Transmit

class HST(nn.Module):
    def __init__(self, num_nodes, num_regions, num_graphs, in_dim = 1, in_dim_cluster = 3, out_dim = 12,
                 channels_0 = 32, channels_1 = 128, channels_2 = 256, support_list = [], transmit = None, mf_embedding_size = 3,
                 num_time_steps = 12, Kt=3, K_ta = 8, order_t = 3, K_sa = 8):

        super(HST, self).__init__()

        self.bn = BatchNorm2d(in_dim, affine=False)
        self.bn_cluster = BatchNorm2d(in_dim_cluster, affine=False)
        self.num_nodes = num_nodes
        self.num_regions = num_regions
        self.num_time_steps = num_time_steps
        self.support_list = support_list

        # starting FC
        self.start_conv_0 = nn.Conv2d(in_channels=in_dim, out_channels=channels_0, kernel_size=(1, 1))
        self.start_conv_c_0 = nn.Conv2d(in_channels=in_dim_cluster, out_channels=channels_0, kernel_size=(1, 1))

        # temporal embedding
        self.conv_tem = nn.Conv2d(in_channels=295, out_channels=channels_0, kernel_size=(1, 1), bias=True)

        # spatial embedding
        self.conv_spa = nn.Conv2d(in_channels=64*num_graphs, out_channels=channels_0, kernel_size=(1, 1), bias=True)

        # micro block
        d_ta = int(channels_0 // K_ta)
        self.micro_block_1 = micro_block(2*channels_0, channels_0, channels_0, Kt, K_ta, d_ta, True, num_graphs, 2, order_t)
        self.micro_block_2 = micro_block(2*channels_0, channels_0, channels_0, Kt-1, K_ta, d_ta, True, num_graphs, 2, order_t)

        # macro block
        d_sa = int(channels_0 // K_sa)
        self.macro_block_1 = macro_block(channels_0, channels_0, channels_0, channels_0, Kt, K_sa, d_sa, K_ta, d_ta, True)
        self.macro_block_2 = macro_block(channels_0, channels_0, channels_0, channels_0, Kt-1, K_sa, d_sa, K_ta, d_ta, True)

        # transmit module
        self.transmit_0 = Transmit(channels_0, channels_0, mf_embedding_size, num_time_steps, transmit, num_nodes, num_regions)
        self.transmit_1 = Transmit(channels_0, channels_0, mf_embedding_size, num_time_steps-(Kt*2-2), transmit, num_nodes, num_regions)
        self.transmit_2 = Transmit(channels_0, channels_0, mf_embedding_size, num_time_steps-(Kt*2), transmit, num_nodes, num_regions)

        #skip conv
        self.skip_conv = nn.Conv2d(in_channels=2*channels_0,
                                    out_channels=channels_1,
                                    kernel_size=(1, 1),
                                    bias=True)

        # ending FC
        self.end_conv_1 = nn.Conv2d(in_channels=channels_1,
                                    out_channels=channels_2,
                                    kernel_size=(1, num_time_steps-(Kt*2)),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=channels_2,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, input_cluster, tem_embedding, spa_embedding, mean, std):
        #preprocessing
        batch_size = input.shape[0]
        x = input
        x_c = input_cluster
        x = self.start_conv_0(x)
        x_c = self.start_conv_c_0(x_c)

        transmit_0 = self.transmit_0(x, x_c)
        x_fusion = (torch.einsum('bmn,bcnt->bcmt', transmit_0, x_c))
        x = torch.cat((x, x_fusion), dim=1)

        #temporal and spatial embedding
        tem_em = torch.unsqueeze(tem_embedding, dim=2)
        tem_em = self.conv_tem(tem_em)
        tem_em_node = tem_em.expand(-1,-1,self.num_nodes,-1)
        tem_em_region = tem_em.expand(-1, -1, self.num_regions, -1)

        spa_em_region = torch.unsqueeze(torch.unsqueeze(spa_embedding, dim=0), dim=0)
        spa_em_region = spa_em_region.permute(0, 3, 2, 1)
        spa_em_region = self.conv_spa(spa_em_region)
        spa_em_region = spa_em_region.expand(batch_size, -1, -1, self.num_time_steps)

        #macro and micro 1
        x = self.micro_block_1(x, self.support_list, tem_em_node)
        x_c = self.macro_block_1(x_c, tem_em_region, spa_em_region)
        transmit_1 = self.transmit_1(x, x_c)
        x_fusion = (torch.einsum('bmn,bcnt->bcmt', transmit_1, x_c))
        x = torch.cat((x, x_fusion), dim=1)

        # macro and micro 2
        x = self.micro_block_2(x, self.support_list, tem_em_node)
        x_c = self.macro_block_2(x_c, tem_em_region, spa_em_region)
        transmit_2 = self.transmit_2(x, x_c)
        x_fusion = (torch.einsum('bmn,bcnt->bcmt', transmit_2, x_c))
        x = torch.cat((x, x_fusion), dim=1)
        x = F.relu(self.skip_conv(x))

        # forecasting block
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x * std + mean
        x = F.relu(x)

        return x
