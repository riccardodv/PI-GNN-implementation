# Import required packages
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import numpy as np
import networkx as nx
import pandas as pd
import os
import torch.nn.functional as F
from collections import OrderedDict
from more_itertools import windowed
os.getcwd()



# 1) crea data loader e prova a vedere trainando parecchi grafi se si impara un modello che funziona direttamente bene
# su un nuovo grafo non visto in precedenza



# Define two−layer GCN with possibility of STE
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_feats, hidden, classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden, norm='right', bias = False)
        self.linear1 = nn.Linear(in_feats, hidden, bias = False)
        self.bn1 = nn.BatchNorm1d(hidden)
        # self.conv2 = GraphConv(hidden, hidden, norm='right', bias = False)
        # self.linear2 = nn.Linear(hidden, hidden, bias = False)
        # self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = GraphConv(hidden, classes, norm='right', bias = False)
        self.linear3 = nn.Linear(hidden, classes, bias = False)
        self.bn3 = nn.BatchNorm1d(classes)
        self.ste = StraightThroughEstimator()

    def forward(self, g, inputs, ste):
        h = self.conv1(g,inputs) + self.linear1(inputs)
        h = self.bn1(h)
        h = torch.relu(h)
        # h = self.conv2(g,h) + self.linear2(h)
        # h = self.bn2(h)
        # h = torch.relu(h)
        h = self.conv3(g,h) + self.linear3(h)
        h = self.bn3(h)
        if ste:
            h = self.ste(h)
        else:
            h = torch.sigmoid(h)

        return h


# Define multi−layers GCN to tackle Gsets
class GCN_dev(nn.Module):
    def __init__(self, in_feats, hidden_sizes, dropout, num_classes):
        super(GCN_dev, self).__init__()
        # Combine all layer sizes into a single list
        all_layers = [in_feats] + [hidden_sizes] + [num_classes]
        # slice list into sub−lists of length 2
        self.layer_sizes = list(windowed(all_layers,2))
        # reference to ID final layer
        self.out_layer_id = len(self.layer_sizes) - 1
        self.dropout_frac = dropout
        self.layers = OrderedDict()
        for idx, (layer_in, layer_out) in enumerate(self.layer_sizes):
            self.layers['idx'] = GraphConv(layer_in , layer_out)

    def forward(self, g, inputs):
        for k, layer in self.layers.items():
            if k == 0: # reference to ID final layer???
                h = layer(g, inputs)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
            elif 0 < k < self.out_layer_id : # intermediate layers
                h = layer(g, h)
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
            else: # output layer
                h = layer(g, h)
                h = torch.sigmoid(h)
            return h


# Define custom loss function for QUBOs
def loss_func(probs_, Q_mat):
      """compute cost value for given soft assignments and predefined QUBO matrix

      """
      # minimize cost=x.T∗Q∗x
      cost = (probs_.T @ Q_mat @ probs_).squeeze()
      # cost = (probs_.T @ Q_mat @ probs_ + torch.sum(probs_)).squeeze() # a regularizer L1 or L2 does not seem to help
      return cost


def reg_graph(z, n, d0):
    """generate the random graph and QUBO matrix

    Returns
    -------
    g
        z-regular graph of n nodes with node-features of dimension d0
    Q
        QUBO matrix of MAX-CUT for this graph
    """
    G = nx.random_regular_graph(z, n)
    assert not G.is_directed()
    m = G.number_of_edges()
    n = G.number_of_nodes()
    G = nx.relabel.convert_node_labels_to_integers(G)
    src = [u for u, v in G.edges()]
    dst = [v for u, v in G.edges()]
    g = dgl.graph(([], []))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    #g = dgl.to_bidirected(g)
    g.ndata["feat"] = torch.randn(n, d0)
    Q = dgl.khop_adj(g, 1) - torch.diag(g.in_degrees())
    return g, Q


def read_graph_from_file(d0, name = './G14', n = 800, Spectral = False):
    """read from Gset a graph - default is G14

    Returns
    -------
    g
        Gset with node-features of dimension d0
    Q
        QUBO matrix of MAX-CUT for this graph
    """
    file_ = pd.read_csv(name, sep=' ')
    src = torch.tensor(file_['src'][1:].values)
    dst = torch.tensor(file_['dst'][1:].values)
    g = dgl.graph(([], []))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.remove_nodes(0)
    if Spectral:
        g = init_ndata_spectral(g, n, d0)
    else:
        g.ndata["feat"] = torch.randn(n, d0)
    Q = dgl.khop_adj(g, 1) - torch.diag(g.in_degrees())
    return g, Q

def init_ndata_spectral(g, n, d0):
    L = torch.diag(g.in_degrees())- dgl.khop_adj(g, 1)
    e, v = torch.eig(L, eigenvectors=True)
    g.ndata["feat"] = v[:,:d0]*10000
    return g


def train(g, Q_mat, model, epochs = 100000, lr= 0.0001, T = 1, ScheduleAnnealing = False, alpha = 0.999, Langevin = False, max_ = True, ste = False):
    """train the model using DGL

    Returns
    -------
    g
        graph with added node features such as the soft assingments 'probs'
        and the binary decision variables 'bdv'
    """
    optimizer = torch.optim.Adam(model.parameters(), lr)
    features = g.ndata['feat']
    # print(features)
    ############
    # all_cuts = []
    ##########
    count = 0
    for e in range(epochs):
        # Forward
        g.ndata['probs'] = model(g, features, ste)
        g.ndata['bdv'] = torch.round(g.ndata['probs'])
        ##########
        # all_cuts.append(count_cuts(g).item())
        # print('counts old and new:', count, count_cuts(g))
        ##########

        # choose to keep max number of cuts up to now or number of cuts in this loop
        if max_:
            count = max(count, count_cuts(g))
        else:
            count = count_cuts(g)

        # Compute loss
        loss = loss_func(g.ndata['probs'], Q_mat)

        # print some values
        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, cuts: {}, polarization: {}'.format(e, loss, count, compute_polarization(g)))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # add Langevin white noise
        if Langevin:
            if ScheduleAnnealing:
                T = T*alpha
                for p in model.parameters():
                    p.grad += torch.randn(p.shape)*np.sqrt(lr)*T
            else:
                for p in model.parameters():
                    p.grad += torch.randn(p.shape)*np.sqrt(lr)*T

        optimizer.step()

    return g, count

def compute_polarization(g):
    p = g.ndata['probs']
    return torch.mean((2*p-1)**2)

def count_cuts(g):
    """counts cuts in a graph

    Returns
    -------
    cuts_1
        number of cuts for the given graph g
    """
    # # Count number of cut edges by checking for all edges if they are cut or not
    # src = g.edges()[0]
    # dst = g.edges()[1]
    # edata_ = []
    # for i in range(len(src)):
    #     cuts = 1 if (g.ndata['bdv'][src[i]] != g.ndata['bdv'][dst[i]]) else 0
    #     edata_.append(cuts)
    # g.edata['cut'] = torch.tensor(edata_).reshape(-1,1)
    # cuts_1 = g.edata['cut'].sum()/2
    # Count cuts using (7) in "Combinatorial Optimization with Physics-Inspired
    # Graph Neural Networks"
    cuts_2 = -1/2*torch.sum(dgl.khop_adj(g, 1)* (2*g.ndata['bdv'] @
                g.ndata['bdv'].T - g.ndata['bdv'].T - g.ndata['bdv']))
    # assert cuts_1 == cuts_2
    return cuts_2.item()


# generate random regular graph and compute cuts with PI-GNN
z = 3; n = 100; d0 = int(np.power(n, 1/3))
g, Q_mat = reg_graph(z, n, d0)
d1 = int(d0/2); d2 = 1
model = GCN(in_feats = d0, hidden =  d1, classes = d2)
g, max_cut = train(g, Q_mat, model, epochs = 1000, lr = 0.0001)
max_cut



# read G14 and compute cuts with PI-GNN
n = 800; d0 = 369 #int(np.power(n, 1/3))
torch.manual_seed(14)
g, Q_mat = read_graph_from_file(d0, Spectral = True)
d1 = 5 #int(d0/2)
d2 = 1
model = GCN(in_feats = d0, hidden =  d1, classes = d2)
g, max_cut = train(g, Q_mat, model, epochs = 10000, lr = 0.00467, T = 100, ScheduleAnnealing = False, alpha = 0.999, Langevin = False, ste = False, max_=False)
print('max_cut', max_cut)




# import multiprocessing
# print(multiprocessing.cpu_count())
