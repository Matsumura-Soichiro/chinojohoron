import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Net(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, embed_dim, class_num):
        super(Net, self).__init__()

        self.conv1 = GraphConv(feature_dim, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=0.8)
        self.conv2 = GraphConv(hidden_dim, embed_dim)
        self.pool2 = TopKPooling(embed_dim, ratio=0.8)
        self.conv3 = GraphConv(embed_dim, embed_dim)
        self.Unpool1 = Unpool()
        self.conv4 = GraphConv(embed_dim, hidden_dim)
        self.Unpool2 = Unpool()
        self.conv5 = GraphConv(hidden_dim, feature_dim)

        self.lin1 = torch.nn.Linear(feature_dim, class_num)

    def forward(self, x, g):
        xs = []
        perms = []
        edge_indexes = []

        x, edge_index, batch = x, g, 1

        edge_indexes.append(edge_index)
        x = F.relu(self.conv1(x, edge_index))
        xs.append(x)
        x, edge_index, _, batch, perm, _ = self.pool1(x, edge_index)
        perms.append(perm)

        edge_indexes.append(edge_index)
        x = F.relu(self.conv2(x, edge_index))
        xs.append(x)
        x, edge_index, _, batch, perm, _ = self.pool2(x, edge_index)
        perms.append(perm)

        x = F.relu(self.conv3(x, edge_index))

        x = self.Unpool1(x, xs[1], perms[1])
        x = x.add(xs[1])
        x = F.relu(self.conv4(x, edge_indexes[1]))

        x = self.Unpool2(x, xs[0], perms[0])
        x = x.add(xs[0])
        x = F.relu(self.conv5(x, edge_indexes[0]))

        x = F.log_softmax(self.lin1(x), dim=-1)

        return x

    def embed(self, x, g):
        x, edge_index, batch = x, g, 1
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool1(x, edge_index)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool2(x, edge_index)
        x = torch.sigmoid(x)
        dim = x.shape[0]
        return torch.sum(x, 0) / dim


class Unpool(torch.nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    def forward(self, x_down, x_up, perm):
        # x, edge_index, batch = x, g, 1
        new_x = x_up.new_zeros([x_up.shape[0], x_down.shape[1]])
        new_x[perm] = x_down
        return new_x
