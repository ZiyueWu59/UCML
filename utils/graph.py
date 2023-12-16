import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn import Parameter

import math

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, num_classes=16, in_channel=300, embed_size=512, norm_func='sigmoid'):
        super(GCN, self).__init__()

        self.num_classes = num_classes
        self.gc1 = GraphConvolution(in_channel, embed_size // 2)
        self.gc2 = GraphConvolution(embed_size // 2,  embed_size)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inp, adj):

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        concept_feature = x
        concept_feature = l2norm(concept_feature)

        return concept_feature


def l2norm(input, axit=-1):
    norm = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-12
    output = torch.div(input, norm)
    return output

class KGVRL(nn.Module):
    def __init__(self, input_dim=512, label_dim=300, feat_dim=512, concept_num=167, sim_matrix=None, all_emb=None):
        super().__init__()

        self.concept_num = concept_num
        self.gat = KGAG(feat_dim, feat_dim, feat_dim, dropout=0.1, alpha=0.1, nheads=8)

        self.joint_fc = nn.Linear(input_dim + label_dim, feat_dim)
        self.text_fc = nn.Linear(label_dim, feat_dim)
        self.sim_matrix = nn.Parameter(sim_matrix, requires_grad=True)
        self.all_emb = nn.Parameter(all_emb, requires_grad=False) # (169, 300)
        self.attn_d = math.sqrt(feat_dim)
        self.layer_normal = nn.LayerNorm(feat_dim)


    def forward(self, v_node, idx):

        t_label = self.all_emb[idx]
        joint_node = torch.cat([v_node, t_label], dim=-1) 
        joint_node = F.normalize(self.joint_fc(joint_node))

        all_emb = F.normalize(self.text_fc(self.all_emb))
        concat_node = torch.cat([all_emb, joint_node], dim=0)
        sim_matrix = self.pad_matrix(v_node, idx)
        output = self.gat(concat_node, sim_matrix)
        
        return output[all_emb.size(0):]

    def pad_matrix(self, v_node, idx):
        lens = idx.size(0)
        sim_matrix = self.sim_matrix
        extend_matrix = F.pad(sim_matrix, [0, lens, 0, lens], mode='constant', value=10e-4)
        extend_matrix[self.concept_num:,:self.concept_num] = sim_matrix[idx]
        extend_matrix[:self.concept_num,self.concept_num:] = sim_matrix[:,idx]
        sub_sim = F.softmax(F.cosine_similarity(v_node.unsqueeze(0), v_node.unsqueeze(1), dim=-1), dim=-1)
        extend_matrix[self.concept_num:,self.concept_num:] = sub_sim
        return extend_matrix

class KGAG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads=8, softmax=False):

        super(KGAG, self).__init__()
        self.dropout = dropout
        self.softmax = softmax

        self.attentions = nn.Sequential(*[AGLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])

        self.out_att = AGLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        if self.softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

class AGLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(AGLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, h, adj):

        Wh = torch.mm(h, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e).to(e.device)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
