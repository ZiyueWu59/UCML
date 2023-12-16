import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import math
from scipy.optimize import linear_sum_assignment
from utils.categories_v2 import vidvrd_CatId2name, vidvrd_CatName2Id

class CrossAlignment(nn.Module):
    def __init__(self, enti_dim, concept_dim, n_head, ffn_dim,
                 EntiNameEmb_path=None, PredNameEmb_path=None):
        super().__init__()

        self.enti_dim = enti_dim
        self.n_head = n_head
        self.concept_dim = concept_dim
        self.ffn_dim = ffn_dim

        self.EntiNameEmb_path = EntiNameEmb_path
        EntiNameEmb = np.load(self.EntiNameEmb_path)
        EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
        self.EntiNameEmb = nn.Parameter(EntiNameEmb, requires_grad = False)  

        self.PredNameEmb_path = PredNameEmb_path
        PredNameEmb = np.load(self.PredNameEmb_path)
        PredNameEmb = torch.from_numpy(PredNameEmb).float()
        self.PredNameEmb = nn.Parameter(PredNameEmb, requires_grad = False) 

        self.TextEmb_fc = MLP(self.concept_dim, self.enti_dim, self.ffn_dim, num_layers=2)
        self.v_RAIM = RAIM(enti_dim, ffn_dim, is_conf=True)
        self.t_RAIM = RAIM(enti_dim, ffn_dim, is_conf=False)

        self.concept_fc = copy.deepcopy(self.TextEmb_fc)

        cat_num = len(vidvrd_CatName2Id)

        self.traj_classifier = nn.Sequential(
            MLP(self.ffn_dim, self.ffn_dim, cat_num, num_layers=3),
            nn.Sigmoid()
        )
        

    def forward(self, enti_node, is_train=True, gt_graph=None):
        if len(enti_node.size()) < 2:
            enti_node.unsqueeze_(0)
        if is_train and gt_graph is not None:
            adj_matrix = gt_graph.adj_matrix.permute(0,2,1) 
            adj_1 = adj_matrix[0]
            adj_2 = adj_matrix[1].permute(1,0)
            adj = torch.matmul(adj_1, adj_2) 
            adj[adj>0] = 1

            gt_pred_ids = gt_graph.pred_cat_ids
            gt_traj_ids = gt_graph.traj_cat_ids 
            
            PredNameEmb = F.normalize(self.TextEmb_fc(self.PredNameEmb))
            PredNameEmb_1 = PredNameEmb[gt_pred_ids]
            PredNameEmb = torch.cat([PredNameEmb[:1], PredNameEmb_1], dim=0)

            EntiNameEmb = F.normalize(self.concept_fc(self.EntiNameEmb))
            concept_emb = EntiNameEmb[gt_traj_ids]

            concept_emb_1 = self.t_RAIM(concept_emb.unsqueeze(0), PredNameEmb, PredNameEmb_1, conf_scores=adj)

            enti_node_1 = self.v_RAIM(enti_node.unsqueeze(0), PredNameEmb, PredNameEmb_1)

            if len(enti_node_1.size()) < 2:
                enti_node_1 = enti_node_1.unsqueeze(0)

            enti_num = enti_node_1.size(0)
            con_num = concept_emb_1.size(0)
            
            if enti_num >= con_num:

                max_num = enti_num
                repeat_time = enti_num // con_num
                sub_num = enti_num % con_num   
                gt_traj_ids_expand = torch.cat([gt_traj_ids.repeat(repeat_time), gt_traj_ids[:sub_num]], dim=-1)       
                concept_emb_1 = torch.cat([concept_emb_1.repeat(repeat_time, 1), concept_emb_1[:sub_num]], dim=0)
                sim_scores = F.cosine_similarity(enti_node_1.unsqueeze(1).repeat(1,max_num,1), concept_emb_1.unsqueeze(0).repeat(max_num,1,1), dim=-1) # (concept_num, enti_num)
            else:
                gt_traj_ids_expand = gt_traj_ids
                sim_scores = F.cosine_similarity(enti_node_1.unsqueeze(1).repeat(1,con_num,1), concept_emb_1.unsqueeze(0).repeat(enti_num,1,1), dim=-1) # (concept_num, enti_num)

            _sim_scores = 1 - sim_scores.detach().cpu().numpy()

            enti_idx, con_idx = linear_sum_assignment(_sim_scores)

            pseudo_label = gt_traj_ids_expand[con_idx]

            concept_emb_1 = concept_emb_1[con_idx]

            cls_label = self.traj_classifier(enti_node_1)

            return pseudo_label, cls_label, enti_node_1, concept_emb_1

        else:
            PredNameEmb = F.normalize(self.TextEmb_fc(self.PredNameEmb))
            PredNameEmb_1 = PredNameEmb[1:]
            enti_node_1 = self.v_RAIM(enti_node, PredNameEmb, PredNameEmb_1)
            if len(enti_node_1.size()) < 2:
                enti_node_1 = enti_node_1.unsqueeze(0)

            cls_label = self.traj_classifier(enti_node_1)
            return cls_label, enti_node_1

def cls_to_label(cls_label):
    # ccls_label: (num_enti, num_cat)
    if len(cls_label.size()) < 2:
        cls_label = cls_label.unsqueeze(0)
    max_score, max_idx = torch.max(cls_label, dim=1)
    return max_score, max_idx
    
def concate_node_function(x):
    if len(x.size()) < 2:
        x = x.unsqueeze(0)
    N = x.size(0)
    x_r = x.unsqueeze(1).repeat(1, N, 1)
    x_c = x.unsqueeze(0).repeat(N, 1, 1)
    con_x = torch.cat([x_r, x_c], dim=-1)
    return con_x


class RAIM(nn.Module):
    def __init__(self, enti_dim, ffn_dim, is_conf=False):
        super().__init__()

        self.enti_dim = enti_dim
        self.ffn_dim = ffn_dim
        self.is_conf = is_conf

        self.fc_alpha = MLP(2*self.enti_dim, self.enti_dim, self.ffn_dim, num_layers=2)
        self.fc_s = copy.deepcopy(self.fc_alpha)
        self.fc_o = copy.deepcopy(self.fc_alpha)
        self.gru = nn.GRU(self.ffn_dim, self.ffn_dim, 1)
        if is_conf:
            self.conf_fc = nn.Sequential(
                MLP(2*self.enti_dim, self.enti_dim, 1, num_layers=2),
                nn.Sigmoid()
            )

    def forward(self, feature_node, PredNameEmb, PredNameEmb_1, iter=1, conf_scores=None):
        if len(feature_node.size()) > 2:
            num_node = feature_node.size(1)
        else:
            num_node = feature_node.size(0)
        for i in range(iter):
            concate_node = concate_node_function(feature_node.squeeze(0))
            if conf_scores is None:
                conf_scores = self.conf_fc(concate_node) 
            elif not self.is_conf:
                conf_scores = conf_scores.unsqueeze(2)
            else:
                print("Error Conf Scores.")
                exit()
            attn_weights = torch.matmul(self.fc_alpha(concate_node), PredNameEmb_1.transpose(0,1)) / math.sqrt(self.ffn_dim)
            attn_scores = F.softmax(attn_weights, dim=-1)

            attn_scores_all = torch.cat([1-conf_scores, conf_scores * attn_scores], dim=-1) 
            attn_pred = torch.matmul(attn_scores_all, PredNameEmb)

            conf_scores_norm = F.normalize(conf_scores, dim=1)
            feature_node_1 = feature_node.repeat(num_node,1,1)

            mask_ij = torch.ones([feature_node_1.size(0), feature_node_1.size(0)]).to(feature_node_1.device)
            mask_ij = (mask_ij - torch.diag_embed(torch.diag(mask_ij))).unsqueeze(2)
            mask_ji = mask_ij.permute(1,0,2)
            memory_ij = conf_scores_norm * self.fc_s(torch.cat([feature_node_1, attn_pred], dim=-1)) * mask_ij
            memory_ji = conf_scores_norm.permute(1,0,2) * self.fc_o(torch.cat([feature_node_1.permute(1,0,2), attn_pred.permute(1,0,2)], dim=-1)) * mask_ji
            memory = torch.sum(memory_ij + memory_ji, dim=1).unsqueeze(0)

            if len(feature_node.size()) < 3:
                feature_node = feature_node.unsqueeze(0)
            update_feature_node, _ = self.gru(feature_node, memory)   

        return update_feature_node.squeeze()   

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
