# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 18:00:44 2021

@author: huymai
"""
import numpy as np
import pandas as pd
import json
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import DataLoader
from torch.nn import Parameter
import torch.nn.init as init
from torch.nn.modules.module import Module 
import torch_geometric

from torch.utils import mkldnn as mkldnn_utils
from torch_scatter import scatter_mean, scatter_max

from sklearn.metrics import f1_score

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# treeGraph stuff
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.feature = []
        self.parent = None
        
def construct_tree(graph):
    index2node = {}
    for i in graph:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in graph:
        indexC = j
        indexP = graph[j]['parent']
        nodeC = index2node[indexC]
        nodeC.feature = graph[j]['vec']
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            root = nodeC
            rootindex = indexC
    root_feature = graph[rootindex]['vec']
    
    # convert graph to adjacency matrix and edge matrix
    adj_matrix = np.zeros([len(index2node), len(index2node)])
    row = []
    col = []
    x_x = []
    edge_matrix = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i].children != None and index2node[index_j] in index2node[index_i].children:
                adj_matrix[index_i][index_j] = 1
                adj_matrix[index_j][index_i] = 1
                row.extend([index_i, index_j])
                col.extend([index_j, index_i])
        x_x.append(index2node[index_i].feature)
    edge_matrix.append(row)
    edge_matrix.append(col)
        
    return x_x, adj_matrix, root_feature, rootindex, edge_matrix, index2node

# Graph Dataset
class GraphDataset(Dataset):
    def __init__(self, list_of_features, edge_indices, labels):
        self.list_of_features = list_of_features
        self.edge_indices = edge_indices
        self.labels = labels

    def __len__(self):
        return len(self.edge_indices)

    def __getitem__(self, index):
        # get edge_index
        edge_index = np.array(self.edge_indices[index])
        
        # get features (x)
        features = self.list_of_features[index]
        
        # get label (y)
        label = self.labels[index]
        
        return Data(x = torch.tensor(features), edge_index = torch.LongTensor(edge_index), y = torch.tensor(label))

# Hyperbolic math
def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
    
# DenseAtt
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to('cpu'), att_adj)
        return att_adj
    
# Manifold
class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
    
class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.
    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K
    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y.float(), p=2, dim=1, keepdim=True) ** 2 
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        x = x.float()
        # print(x.dtype)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        hidden_dim = 32
        if x.shape[0] == hidden_dim and len(x.shape) == 1:
            # print(x.shape)
            x = torch.reshape(x, (1, hidden_dim))
        K = 1. / c
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        # print(vals.shape)
        # print("====")
        # print(x.shape)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x.float(), p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c).float()
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
    
# Hyperbolic layers
def get_dim_act_curv(c, act, num_layers, feat_dim, dim):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    acts = [act] * (num_layers - 1)
    dims = [feat_dim] + ([dim] * (num_layers - 1))
    n_curvatures = num_layers - 1
        
    # fixed curvature
    curvatures = [torch.tensor([c]) for _ in range(n_curvatures)]
    curvatures = [curv.to(device) for curv in curvatures]
        
    return dims, acts, curvatures

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        h = self.hyp_act(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)   # manifold
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)   # DenseAtt

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att.to('cpu'), x_tangent.to('cpu'))
        else:
            support_t = torch.spmm(adj.to('cpu'), x_tangent.to('cpu')).to('cpu')
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = nn.LeakyReLU()

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = torch.tensor(xt)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    
# Encoders
class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers(input)
        else:
            output = self.layers(x)
        return output
    
class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, dropout, bias, use_att, local_agg):
        super(HGCN, self).__init__(c)
        dims, acts, self.curvatures = get_dim_act_curv(c=c, act='relu', num_layers=2, feat_dim=128, dim=32 )
        self.manifold = Hyperboloid()
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, dropout, act, bias, use_att, local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])   # manifold
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super().encode(x_hyp, adj)

def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

# Base Model
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, device, c):
        super(BaseModel, self).__init__()
        
        # c is the hyperbolic curvature
        self.c = torch.tensor([c]).to(device)
       
        self.encoder = HGCN(c=self.c, dropout=0.5, bias=1, use_att=True, local_agg=False)
        
        
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        

    def forward(self, x, edge_m, batch):
        o = torch.zeros_like(x)

        # derive adj_matrix from edge_m
        edge_m_transposed = torch.transpose(edge_m, 0, 1)
        adj = torch.zeros_like(torch.empty(len(x), len(x)))
        for item in edge_m_transposed:
            adj[item[0]][item[1]] = 1
          
        h = self.encoder.encode(x, adj)
        # take care of the row of nans
        h[torch.isnan(h)] = 0
        # print(h)
        
        # h = F.normalize(h, dim=0)
        
        # mean pooling
        h = scatter_max(h, batch.batch.to('cpu'), dim=0)
        # print(h)
        # h = torch_geometric.nn.pool.avg_pool_x(torch.zeros(h.shape[0]).to('cuda'), h, torch.zeros(h.shape[0]).to('cuda'))
        
        # normalize output of mean pooling
        h = F.normalize(h[0], dim=0)
        # print(h)
        out = self.output(h)
        return out

# Base Model 2

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class BaseModelH(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BaseModelH, self).__init__()
        self.gcn1 = GCNConv(in_feats, hid_feats)
        self.gcn2 = GCNConv(hid_feats, out_feats)
    
    def forward(self, data):
        x1 = self.gcn1(data['x'].float(), data['edge_index'].float())
        x2 = self.gcn2(x1, data['edge_index'].float())
        return x2
    
def update_function(param, grad, loss, learning_rate):
  return param - learning_rate * grad

# main code
updated_df = pd.read_csv('./data/updated_pheme_rnr1.csv').drop(['Unnamed: 0'], axis=1)
print(updated_df)

# updated_df = updated_df.astype({'parent': str})

treeDic = {}

for index, row in updated_df.iterrows():
    if math.isnan(row['parent']):
        root_id = row['graph_index']
        
        if not treeDic.__contains__(root_id):
            treeDic[root_id] = {}
            
        indexP = 'None'
        
    else:
        # hardcode...
#         if index == 30:
#             indexP = 12
#         elif index in range(126, 140):
#             indexP = 0
#         elif index in range(194, 208):
#             indexP = 0
#         elif index in range(303, 308) or index in range(309, 316):
#             indexP = 0
#         elif index == 318:
#             indexP = 1
#         elif index == 356:
#             indexP = 8
#         elif index in range(389,393) or index in range(395,408):
#             indexP = 0
#         else:
        indexP = updated_df.loc[updated_df['id'] == row['parent'], 'node_index'].values[0]
            
    indexC = row['node_index']
    vec = json.loads(row['word_embedding'])
    vec = np.array(vec)
    treeDic[root_id][indexC] = {'parent': indexP, 'vec': vec}
    
labelDic = {}
for index, row in updated_df.iterrows():
    if row['node_index'] == 0:
        root_id = row['graph_index']
        
        if row['label'] == 'rumours':
            label = 1
        else:
            label = 0
        
        if not labelDic.__contains__(root_id):
            labelDic[root_id] = label
            
list_of_features = []
edge_indices = []
labels = []

for i in range(1, 6426):
    feat, graph, rootFeat, rootIndex, edge_m, index2node = construct_tree(treeDic[i])
    
    list_of_features.append(feat)
    edge_indices.append(edge_m)
    labels.append(labelDic[i])
    
dataset = GraphDataset(list_of_features, edge_indices, labels)

batch_size = 1
test_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = 1
model = BaseModel(device, c).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.BCELoss()
epochs = 5 

# train model
print("=====Training model=====")
print(list(model.parameters()))
running_train_losses = []
model.train()
for epoch in range(epochs):
    correct = 0
    for num, batch in enumerate(train_loader):
        
        model.zero_grad()
        optimizer.zero_grad()
        if num % 1000 == 0:
            print("+++++++++")
            print(list(model.parameters()))
        output = model(batch['x'].to(device), batch['edge_index'].to(device), batch)
        loss = loss_fn(output, torch.reshape(batch['y'], (1,1)).float().to(device))
        loss.backward()
        optimizer.step()
        
        # with torch.no_grad():
        #     for p in model.parameters():
        #         new_val = update_function(p, p.grad, loss, 0.1)
                
        if num % 1000 == 0:
            print(list(model.parameters()))
            print("+++++++++")
        
        # if output.item() >= threshold:
        #     predicted = torch.LongTensor([[1]]).to('cuda')
        # else:
        #     predicted = torch.LongTensor([[0]]).to('cuda')
        
        predicted = torch.round(output.data).long()
        
        if predicted == torch.reshape(batch['y'], (1,1)).to(device):
            correct += 1
                  
        # if num % 250 == 0:
        #     print(output)
        #     print(batch['y'])
            
        if num % 1000 == 0:
            print(num)
            print("Loss: ", loss.item())
            
        running_train_losses.append(loss.item())
        
    print("Epoch %i complete! Average loss was %.4f" % (epoch + 1, sum(running_train_losses) / len(running_train_losses)))
    print("Train accuracy: " , 100.0 * (correct / 5140))
            
# evaluate model
print("====EVALUATING MODEL====")
correct = 0
y_true = []
y_pred = []
with torch.no_grad():
    for num, batch in enumerate(test_loader):
        
        output = model(batch['x'].to(device), batch['edge_index'].to(device), batch)
        predicted = torch.round(output.data).long()
        
        # if output.item() >= threshold:
        #     predicted = torch.LongTensor([[1]]).to('cuda')
        # else:
        #     predicted = torch.LongTensor([[0]]).to('cuda')
        
        if predicted == torch.reshape(batch['y'], (1,1)).to(device):
            correct += 1
            
        y_true.append(torch.reshape(batch['y'], (1,1)).to(device).item())
        y_pred.append(predicted.item())
            
test_acc = 100.0 * (correct / 1285)
f1 = f1_score(y_true, y_pred, average='binary')
print("Test accuracy: ", test_acc)
print("F1 Score: ", 100.0 * f1)
    