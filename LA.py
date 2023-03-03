import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LAGCN(torch.nn.Module):
    def __init__(self, concat, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LAGCN, self).__init__()

        self.convs_initial = torch.nn.ModuleList()
        self.bns_initial = torch.nn.ModuleList()
        for _ in range(concat):   # 0 1
            self.convs_initial.append(GCNConv(in_channels, hidden_channels, cached=True))   # 128 128
            self.bns_initial.append(torch.nn.BatchNorm1d(hidden_channels))   # 128

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 2):  # 0
            self.convs.append(
                GCNConv(concat*hidden_channels, concat*hidden_channels, cached=True))   # 2*128  2*128
            self.bns.append(torch.nn.BatchNorm1d(concat*hidden_channels))
        self.convs.append(GCNConv(concat*hidden_channels, out_channels, cached=True))  # 2*128 40

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for conv in self.convs_initial:
            conv.reset_parameters()
        for bn in self.bns_initial:
            bn.reset_parameters()

    def forward(self, X, adj_t):
        # adj_t = adj_t.permute(2, 1, 0)
        hidden_list = []
        for i, conv in enumerate(self.convs_initial):
            x = conv(X, adj_t)
            x = self.bns_initial[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden_list.append(x)
        x = torch.cat((hidden_list), dim=-1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

