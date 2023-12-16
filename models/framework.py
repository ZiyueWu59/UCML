import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import info_nce
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.utils_func import vIoU_ts,dura_intersection_ts,unique_with_idx_nd,stack_with_padding
from utils.categories_v2 import vidvrd_CatId2name,PKU_vidvrd_CatId2name
from utils.cross_alignment import CrossAlignment
from utils.graph import GCN, KGVRL
import pickle
from utils.prompt_utils_glove import PromptLearners as lstm_prompt

def stack_with_repeat_2d(tensor_list,dim):
    assert len(tensor_list[0].shape) == 2
    device = tensor_list[0].device
    shape_list = [t.shape for t in tensor_list]
    num_rows = torch.tensor([sp[0] for sp in shape_list])
    num_cols = torch.tensor([sp[1] for sp in shape_list])

    if torch.all(num_rows == num_rows[0]):
        max_L = num_cols.max()
        repeat_dim=1
    elif torch.all(num_cols == num_cols[0]):
        max_L = num_rows.max()
        repeat_dim=0
    else:
        assert False
    
    after_repeat = []
    for tensor in tensor_list:
        L = tensor.shape[repeat_dim]
        n_pad = L - (max_L % L)
        ones = [1]*max_L
        zeros = [0]*n_pad
        total = torch.tensor(ones + zeros,device=device)
        total = total.reshape(-1,L)
        repeats_ = total.sum(dim=0)
        after_repeat.append(
            tensor.repeat_interleave(repeats_,dim=repeat_dim)
        )
    return torch.stack(after_repeat,dim=dim)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before

    def _get_activation_fn(self,activation):

        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # q == k
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class RoleAttnDecoderLayer(nn.Module):

    def __init__(self, dim_pred, nhead, n_preds,dim_enti,dim_att,dim_ffn,dropout=0.1):
        super().__init__()
        self.dim_pred = dim_pred
        self.dim_enti = dim_enti
        self.num_querys = n_preds
        self.dim_ffn = dim_ffn
        self.dim_att = dim_att
        self.self_attn = nn.MultiheadAttention(dim_pred, nhead, dropout=dropout)
        
        fc_rolewise = nn.Sequential(
            nn.Linear(self.dim_enti,self.dim_pred),
            nn.ReLU(),
            nn.Linear(self.dim_pred,self.dim_pred)
        )
        self.fc_rolewise = _get_clones(fc_rolewise, 2)

        self.fc_enti2att = nn.Linear(self.dim_enti,self.dim_att)
        self.fc_pred2att = nn.Linear(self.dim_pred,self.dim_att)

        self.fc2 = nn.Sequential(
            nn.Linear(dim_pred, dim_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, dim_pred)
        )

        self.norm1 = nn.LayerNorm(dim_pred)
        self.norm2 = nn.LayerNorm(dim_pred)
        self.norm3 = nn.LayerNorm(dim_pred)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,pred_query,pos_emb,enco_output):
        
        v = pred_query[:,None,:]
        q = k = self.with_pos_embed(v, pos_emb[:,None,:])
        pred_query2 = self.self_attn(k, q, v)[0].squeeze(1) 
        
        pred_query = self.norm1(pred_query + pred_query2)

        pred_query = self.with_pos_embed(pred_query, pos_emb)
        enti2att = self.fc_enti2att(enco_output) 
        pred2att = self.fc_pred2att(pred_query) 

        enti2att_subjobj = (enti2att[:,:self.dim_att//2],enti2att[:,self.dim_att//2:]) 
        pred2att_subjobj = (pred2att[:,:self.dim_att//2],pred2att[:,self.dim_att//2:])  

        att_matrx = []
        for i in range(2):
            enti2att = enti2att_subjobj[i].t()
            pred2att = pred2att_subjobj[i]
            
            att_mat_i = torch.matmul(pred2att,enti2att) / np.sqrt(self.dim_enti)
            att_matrx.append(att_mat_i)
        
        att_matrx = torch.stack(att_matrx,dim=0) 
        att_matrx_enti = torch.softmax(att_matrx,dim=2)
        att_matrx_role = torch.softmax(att_matrx,dim=0)
        att_matrx = att_matrx_enti * att_matrx_role
        
        role_queries = []
        for idx, fc in enumerate(self.fc_rolewise):
            values = torch.matmul(att_matrx[idx,:,:],enco_output)
            role_q = fc(values)  
            role_queries.append(role_q)
        role_queries = torch.stack(role_queries,dim=0).sum(dim=0) 

        pred_query = self.norm2(pred_query + role_queries)

        pred_query2 = self.fc2(pred_query)

        pred_query = self.norm3(pred_query + pred_query2)
        
        return pred_query,att_matrx

class UCML(nn.Module): 

    def __init__(self,config,is_train=True):
        super(UCML, self).__init__()
        self.is_train = is_train

        self.num_pred_cats = config["num_pred_cats"]
        self.num_enti_cats = config["num_enti_cats"]
        self.dim_feat = config["dim_feat"]        
        self.dim_clsme = config["dim_clsme"]
        self.dim_enti = config["dim_enti"]
        self.dim_pred = config["dim_pred"]
        self.dim_att  = config["dim_att"] 
        self.dim_ffn  = config["dim_ffn"]
        
        self.enco_pool_len = config["enco_pool_len"]
        self.n_enco_layers = config["n_enco_layers"]
        self.n_deco_layers = config["n_deco_layers"]
        self.n_att_head = config["n_att_head"]
        self.num_querys = config["num_querys"]
        self.num_anchors = self.num_querys 

        self.bias_matrix_path = config["bias_matrix_path"]
        self.EntiNameEmb_path = config["EntiNameEmb_path"]
        self.PredNameEmb_path = config["PredNameEmb_path"]

        EntiNameEmb = np.load(self.EntiNameEmb_path)
        EntiNameEmb = torch.from_numpy(EntiNameEmb).float()
        self.EntiNameEmb = nn.Parameter(EntiNameEmb, requires_grad = False)  


        self.neg_weight = config["neg_weight"]
        self.loss_factor = config["loss_coeff_dict"]   
        self.cost_factor = config["cost_coeff_dict"]
        self.positive_vIoU_th = config["positive_vIoU_th"]

        self.pos_embedding = nn.Parameter(torch.FloatTensor(self.num_querys,self.dim_pred),requires_grad=True)
        self.pred_query_init = nn.Parameter(torch.FloatTensor(self.num_querys,self.dim_pred),requires_grad=True)
        
        bias_matrix = np.load(self.bias_matrix_path)
        bias_matrix = torch.from_numpy(bias_matrix).float()
        self.bias_matrix = nn.Parameter(bias_matrix, requires_grad = True)  
        assert self.bias_matrix.shape == (self.num_enti_cats,self.num_enti_cats,self.num_pred_cats)

        self.fc_feat2enti = nn.Sequential(
            nn.Linear(self.dim_feat,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )
        self.fc_bbox2enti = nn.Sequential(
            nn.Linear(8,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )
        self.conv_feat2enti = nn.Conv1d(self.dim_enti*2,self.dim_enti,kernel_size=3,padding=1,stride=2)

        self.fc_enti2enco = nn.Sequential(
            nn.Linear(self.dim_enti*self.enco_pool_len,self.dim_enti),
            nn.ReLU(),
            nn.Linear(self.dim_enti,self.dim_enti),
            nn.ReLU()
        )

        encoder_layer = TransformerEncoderLayer(
            self.dim_enti, self.n_att_head, self.dim_ffn,
            dropout=0.1, activation='relu', normalize_before=False
        )
        self.encoder_layers = _get_clones(encoder_layer, self.n_enco_layers)

        decoder_layer = RoleAttnDecoderLayer(
            self.dim_pred,self.n_att_head,self.num_querys,
            self.dim_enti,self.dim_att,self.dim_ffn,dropout=0.1
        )
        self.decoder_layers = _get_clones(decoder_layer, self.n_deco_layers)
        
        self.fc_pred2logits_pre = nn.Linear(self.dim_pred + self.dim_enti*8, self.num_pred_cats)

        self.CMA = CrossAlignment(self.dim_enti, 300, 8, self.dim_ffn, self.EntiNameEmb_path, self.PredNameEmb_path)

        self.loss_nce = info_nce.InfoNCE(temperature=0.1, negative_mode='unpaired')
        self.loss_nce_no_negative = info_nce.InfoNCE(temperature=0.1)
        self.alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad = True)

        self.loss_triplet = TripletLoss(margin=0.1)

        self.gcn = GCN(16, 300, 512)

        relation_path = config['relation_path']
        concept_graph_path = config['concept_graph_path']
        vidvrd_relation = pickle.load(open(relation_path, 'rb'))
        self.AllEmb = nn.Parameter(vidvrd_relation['embedding'], requires_grad=False)
        self.SimMatrix = vidvrd_relation['sim_matrix']
        concept_graph = pickle.load(open(concept_graph_path, 'rb'))
        self.related_idx = []
        self.related_adj = []
        cat_num = len(PKU_vidvrd_CatId2name)
        for k in concept_graph:
            temp = concept_graph[k]
            self.related_idx.append((temp['idx'].unsqueeze(0)))
            self.related_adj.append(temp['adj'].unsqueeze(0))
            if len(self.related_idx) >= cat_num:
                break
            
        self.related_idx = torch.cat(self.related_idx)
        self.related_adj = nn.Parameter(torch.cat(self.related_adj))

        self.lstm_prompt = lstm_prompt()

        self.visual_gat = KGVRL(512, 300, 512, 169, self.SimMatrix, self.AllEmb)

        self._reset_parameters()

    
    def _reset_parameters(self):
        skip_init_param_names = [
            "bias_matrix",
            "EntiNameEmb",
            "pred_query_init",
        ]
        for name,p in self.named_parameters():
            if name in skip_init_param_names:  
                print("skip init param: {}".format(name))
                continue

            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
            if "bias" in name:
                nn.init.zeros_(p)

        nn.init.normal_(self.pred_query_init,mean=0,std=0.1)
        nn.init.normal_(self.pos_embedding,mean=0,std=0.1)
    
    def forward(self,proposal_list,gt_graph_list=None,topk=10):

        if self.is_train:
            assert gt_graph_list != None
            return self._forward_train(proposal_list,gt_graph_list)
        else: 
            self.topk = topk
            return self._forward_test(proposal_list)


    def _forward_test(self,proposal_list):

        triplets = []
        pred_queries = []
        pred_centerspan = []
        pred_regression = []
        xx = []
        for ii,proposal in enumerate(proposal_list):
            if proposal.num_proposals == 0:  
                triplets.append(None)
                pred_queries.append(None)
                pred_regression.append(None)
            else:
                pred_query,pred_logits,att_matrx, cls_label = self.encode2decode(proposal, is_train=False)

                ret = self.construct_triplet(proposal,pred_logits,att_matrx, cls_label)
                triplets.append(ret)

        return triplets


    def _preprocess_proposal(self,proposal):
        video_len,video_wh = proposal.video_len, proposal.video_wh
        w,h = video_wh
        traj_durations,bboxes_list,feature_list = proposal.traj_durations,proposal.bboxes_list,proposal.features_list

        traj_bboxes = []
        traj_features = []
        for pid in range(len(bboxes_list)):
            bboxes = bboxes_list[pid].clone()  
            bboxes[:,0:4:2] /= w  
            bboxes[:,1:4:2] /= h 

            bbox_ctx = (bboxes[:,2] + bboxes[:,0])/2
            bbox_cty = (bboxes[:,3] + bboxes[:,1])/2
            bbox_w = bboxes[:,2] - bboxes[:,0]
            bbox_h = bboxes[:,3] - bboxes[:,1]

            diff_ctx = bbox_ctx[1:] - bbox_ctx[:-1]
            diff_cty = bbox_cty[1:] - bbox_cty[:-1]
            diff_w = bbox_w[1:] - bbox_w[:-1]
            diff_h = bbox_h[1:] - bbox_h[:-1]
            bbox_feat = [
                bbox_ctx,diff_ctx,
                bbox_cty,diff_cty,
                bbox_w,diff_w,
                bbox_h,diff_h
            ]
            bbox_feat = stack_with_padding(bbox_feat,dim=1) 
            traj_bboxes.append(bbox_feat)

            features = feature_list[pid]
            traj_features.append(features)

        
        traj_bboxes = stack_with_repeat_2d(traj_bboxes,dim=0)    
        traj_features = stack_with_repeat_2d(traj_features,dim=0) 

        return traj_bboxes,traj_features,traj_durations



    def encode2decode(self, proposal, is_train=False, gt_graph=None):
        n_trajs,video_len = proposal.num_proposals,proposal.video_len

        traj_bboxes,traj_features,traj_dura = self._preprocess_proposal(proposal) 

        traj_visual  = traj_features[:,:,:self.dim_feat]

        traj_bboxes =   self.fc_bbox2enti(traj_bboxes) 
        traj_visual = self.fc_feat2enti(traj_visual)
        enti_features = torch.cat([traj_bboxes,traj_visual],dim=-1)
        
        enti_features = enti_features.permute(0,2,1) 
        enti_nodes = self.conv_feat2enti(enti_features)
        enti_nodes = enti_nodes.permute(0,2,1) 

        enti2enco = enti_nodes.permute(0,2,1)
        enti2enco = F.adaptive_max_pool1d(enti2enco,output_size=self.enco_pool_len)
        enti2enco = enti2enco.reshape(n_trajs,-1)
        enti2enco = self.fc_enti2enco(enti2enco)  

        if is_train:
            pseudo_label, cls_label, traj_node_features, pseudo_label_features = self.CMA(enti2enco, is_train, gt_graph)
        else:
            cls_label, traj_node_features = self.CMA(enti2enco, is_train)
            _, pseudo_label = self.cls_to_label(cls_label)

        output = enti2enco[:,None,:]
        for layer in self.encoder_layers:
            output = layer(output)
        enco_output = output.squeeze(1) 

        pred_queries = self.pred_query_init
        for layer in self.decoder_layers:
            pred_queries,att_matrx = layer(pred_queries, self.pos_embedding, enco_output)
        
        pred_logits = self.prediction_head(pred_queries, att_matrx, pseudo_label, enti2enco) 

        
        if is_train:
            return pred_queries, pred_logits, att_matrx, pseudo_label, traj_node_features, pseudo_label_features, cls_label
        else:
            return pred_queries, pred_logits, att_matrx, cls_label


    def prediction_head(self,pred_queries,att_matrx,cat_ids,enti_feat):

        pred_soid = torch.argmax(att_matrx,dim=-1) 
        pred_socatid = cat_ids[pred_soid] 
        pred_bias = self.bias_matrix[pred_socatid[0,:],pred_socatid[1,:],:] 
   

        sub_clsme = self.lstm_prompt(pred_socatid[0,:])
        obj_clsme = self.lstm_prompt(pred_socatid[1,:])
        sub_clsme_gcn = self.obtain_gcn_feature(pred_socatid[0,:])
        obj_clsme_gcn = self.obtain_gcn_feature(pred_socatid[1,:])  

        sub_feat = enti_feat[pred_soid[0,:],:] 
        obj_feat = enti_feat[pred_soid[1,:],:] 
        sub_feat_gat = self.visual_gat(sub_feat, pred_socatid[0,:])
        obj_feat_gat = self.visual_gat(obj_feat, pred_socatid[1,:])

        pred_queries = torch.cat([pred_queries,sub_clsme, sub_clsme_gcn, obj_clsme, obj_clsme_gcn, sub_feat, sub_feat_gat, obj_feat, obj_feat_gat],dim=-1)  # shape == (n_querys,dim_pred+600+2*dim_enti)

        pred_logits = self.fc_pred2logits_pre(pred_queries)

        pred_logits = pred_logits + pred_bias
        
        return pred_logits
  
    def obtain_gcn_feature(self, idx, emb=None):

        related_idx = self.related_idx[idx] 
        related_emb = self.AllEmb[related_idx] 
        if emb is not None:
            related_emb[:, 0, :] = emb
        related_adj = self.related_adj[idx] 

        output = self.gcn(related_emb, related_adj)

        return output[:, 0, :] 
  
    def _forward_train(self,proposal_list,gt_graph_list):

        gt_preds = [g.pred_cat_ids for g in gt_graph_list]

        gt_adjs = [g.adj_matrix for g in gt_graph_list]
        gt_adj_enti_align = []

        assert len(proposal_list) == len(gt_graph_list) 

        mp_results = [self.encode2decode(proposal=proposal, is_train=True, gt_graph=gt_graph) for proposal, gt_graph in zip(proposal_list, gt_graph_list)]
        pred_queries,pred_logits,att_matrx, pseudo_label, traj_node_features, pseudo_label_features, cls_label = list(zip(*mp_results))

        
        for gt_adj,proposal,gt_graph in zip(gt_adjs,proposal_list,gt_graph_list):
            gaea,viou_mat = self.enti_viou_align(gt_adj,proposal,gt_graph)
            gt_adj_enti_align.append(gaea)

        indices = []
        for pred_logit,gt_pred,att,gt_adj in zip(pred_logits,gt_preds,att_matrx,gt_adj_enti_align):
            index = self.bipartite_match(
                pred_logit,gt_pred,att,gt_adj
            )
            indices.append(index)


        loss_dict = self.sgg_loss(indices,
            pred_logits,att_matrx,
            gt_preds,gt_adj_enti_align,
        )

        total_loss = torch.stack(list(loss_dict.values())).sum()   

        contrastive_loss = [self.triplet_loss(p_label, enti_node, concept_emb) for \
            p_label, enti_node, concept_emb in zip(pseudo_label, traj_node_features, pseudo_label_features)]
        contrastive_loss = torch.stack(contrastive_loss, 0).sum()
        loss_dict['contrastive loss'] = contrastive_loss


        cls_loss = [self.cls_loss(p_label, c_label) for p_label, c_label in zip(pseudo_label, cls_label)]
        cls_loss = torch.stack(cls_loss, 0).sum()
        loss_dict['classifier_loss'] = cls_loss

        return total_loss + contrastive_loss + cls_loss, loss_dict


    def triplet_loss(self, pseudo_label, enti_node, concept_emb):

        if enti_node.size(0) == 1 and concept_emb.size(0) == 1:
            total_loss = self.loss_nce_no_negative(concept_emb,enti_node)
            return total_loss
        uniq_cat, uniq_index = unique_with_idx_nd(pseudo_label)
        if uniq_cat.size(0) == 1:
            cat_num = uniq_index[0].size(0)
            uniq_index = uniq_index[0].unsqueeze(1)
        else:
            cat_num = uniq_cat.size(0)
        total_loss = []
        for i in range(cat_num):
            positive_idx = uniq_index[i]
            index_list = list(uniq_index)
            index_list.pop(i)
            negative_idx = torch.cat(index_list).squeeze()
            if len(positive_idx.size()) < 1:
                positive_idx = positive_idx.unsqueeze(0)
            if len(negative_idx.size()) < 1:
                negative_idx = negative_idx.unsqueeze(0)
            query = concept_emb[positive_idx] 
            pos_enti = enti_node[positive_idx] 
            neg_enti = enti_node[negative_idx].unsqueeze(1).repeat(1, positive_idx.size(0), 1) 
            tmp_loss = [self.loss_triplet(query, pos_enti, neg_enti[i]) for i in range(neg_enti.size(0))]
            tmp_loss = torch.stack(tmp_loss, 0).mean()
            total_loss.append(tmp_loss)
        total_loss = torch.stack(total_loss, 0).mean()
        return total_loss
    
    def cls_loss(self, psudeo_label, cls_label):
        cls_num = len(PKU_vidvrd_CatId2name)
        enti_num = psudeo_label.size(0)
        target = torch.zeros([enti_num, cls_num], dtype=cls_label.dtype).to(psudeo_label.device)
        for idx in range(enti_num):
            target[idx][psudeo_label[idx]] = 1

        cls_loss = F.binary_cross_entropy(cls_label, target)

        return cls_loss
    
    def enti_viou_align(self,gt_adj,proposal,gt_graph):
        _,n_gt_pred,n_gt_enti = gt_adj.shape
        n_enti = proposal.num_proposals
        gt_adj_enti_align = torch.zeros(size=(2,n_gt_pred,n_enti),device=gt_adj.device)


        pr_trajbboxes,pr_trajduras = proposal.bboxes_list,proposal.traj_durations
        gt_trajbboxes,gt_trajduras = gt_graph.traj_bboxes,gt_graph.traj_durations      
        gt_trajduras[:,1] -= 1 
        num_gt_enti = len(gt_trajbboxes)
        inter_dura,dura_mask = dura_intersection_ts(pr_trajduras,gt_trajduras) 
        
        inter_dura_p = inter_dura - pr_trajduras[:,0,None,None] 
        inter_dura_g = inter_dura - gt_trajduras[None,:,0,None]
        
        pids,gids = dura_mask.nonzero(as_tuple=True) 
        viou_matrix = torch.zeros_like(dura_mask,dtype=torch.float)
        for pid,gid in zip(pids.tolist(),gids.tolist()):
            dura_p = inter_dura_p[pid,gid,:]
            dura_g = inter_dura_g[pid,gid,:]
            bboxes_p = pr_trajbboxes[pid]
            bboxes_g = gt_trajbboxes[gid]
            viou_matrix[pid,gid] = vIoU_ts(bboxes_p,bboxes_g,dura_p,dura_g)
        
        viou_mask = viou_matrix > self.positive_vIoU_th
        maxvIoU_prop_ind = torch.argmax(viou_matrix,dim=0)   
        sum0_mask = viou_mask.sum(dim=0) == 0             
        maxvIoU_prop_ind = maxvIoU_prop_ind[sum0_mask]
        viou_mask[maxvIoU_prop_ind,sum0_mask] = 1           
        assert viou_mask.sum() >= num_gt_enti

        for pid, row in enumerate(viou_mask):
            if torch.sum(row) == 0:
                pass
            else:
                gt_enti_ind = torch.argmax(viou_matrix[pid,:])
                gt_adj_enti_align[:,:,pid] = gt_adj[:,:,gt_enti_ind]
        
        return gt_adj_enti_align,viou_matrix

    @torch.no_grad()
    def bipartite_match(self,
            pred_logit,gt_pred,att_matrx,gt_adj_enti_align,
        ):
        _,n_gt_pred,n_enti = gt_adj_enti_align.shape

        pred_logit = pred_logit[:,:,None].repeat(1,1,n_gt_pred)
        gt_pred = gt_pred[None,:].repeat(self.num_querys,1) 

        cost_cls = F.cross_entropy(pred_logit,gt_pred,reduction='none') 
        

        att_matrx = att_matrx[:,:,None,:].repeat(1,1,n_gt_pred,1)
        gt_adj_enti_align = gt_adj_enti_align[:,None,:,:].repeat(1,self.num_querys,1,1)
        cost_adj = F.binary_cross_entropy(att_matrx,gt_adj_enti_align,reduction='none') 
        cost_adj = cost_adj.mean(dim=[0,-1])  
        
        
        cost_cls *= self.cost_factor["classification"]
        cost_adj *= self.cost_factor["adj_matrix"]
        cost_all = cost_cls + cost_adj
        index = linear_sum_assignment(cost_all.cpu())

        return index


    def sgg_loss(self,indices,
            pred_logits,att_matrx,gt_preds,gt_adj
        ):
        
        gt_targets = []
        for i,(idx_anchor,idx_gt) in enumerate(indices):
            gt_preds_align = torch.zeros(size=(self.num_querys,),dtype=torch.long,device=gt_preds[0].device)
            gt_preds_align[idx_anchor] = gt_preds[i][idx_gt]
            gt_targets.append(gt_preds_align)
        gt_targets = torch.cat(gt_targets)
        pos_mask = gt_targets != 0 
        neg_mask = torch.logical_not(pos_mask)  
        pred_logits_ = torch.cat(pred_logits,dim=0)
        cls_loss = F.cross_entropy(pred_logits_,gt_targets,reduction='none') 
        cls_pos = cls_loss[pos_mask].mean()
        if neg_mask.sum() > 0:
            cls_neg = cls_loss[neg_mask].mean()
        else:
            cls_neg = torch.zeros_like(cls_pos)
        
        att_matrx_ = [att_matrx[i][:,idx_anchor,:].reshape(2,-1) for i,(idx_anchor,_) in enumerate(indices)]  # shape == (2,n_positive,n_enti) --> (2,n_positive*n_enti)
        gt_adj = [gt_adj[i][:,idx_gt,:].reshape(2,-1) for i,(_,idx_gt) in enumerate(indices)]  # shape == (2,n_positive,n_enti)
        att_matrx_ = torch.cat(att_matrx_,dim=-1)
        gt_adj = torch.cat(gt_adj,dim=-1)
        bce_weight = torch.ones_like(gt_adj)
        gt_adj_mask = gt_adj.type(torch.bool)
        bce_weight[~gt_adj_mask] *= self.neg_weight
        adj_loss = F.binary_cross_entropy(att_matrx_,gt_adj,weight=bce_weight,reduction='none').mean()
        
        cls_pos *= self.loss_factor["classification"]
        cls_neg *= self.loss_factor["classification"]
        adj_loss *= self.loss_factor["adj_matrix"]
        loss_dict = {
            "cls_pos":cls_pos,
            "cls_neg":cls_neg,
            "adj":adj_loss
        }
        return loss_dict

    def cls_to_label(self, cls_label):

        if len(cls_label.size()) < 2:
            cls_label = cls_label.unsqueeze(0)
        max_score, max_idx = torch.max(cls_label, dim=1)
        return max_score, max_idx

    def construct_triplet(self, proposal, pred_logits, att_matrx, cls_label):

        
        pred_probs = torch.softmax(pred_logits,dim=-1)
        pred_scores,pred_catids = torch.topk(pred_probs,self.topk,dim=-1) 
        pred_scores = pred_scores.reshape(-1) 
        pred_catids = pred_catids.reshape(-1)
        predquery_ids = torch.tensor(list(range(self.num_anchors)),device=pred_catids.device) 
        predquery_ids = torch.repeat_interleave(predquery_ids,self.topk)
        
        traj_duras = proposal.traj_durations.clone()
        n_traj = traj_duras.shape[0]

        enti_scores, enti_catids = self.cls_to_label(cls_label)
        
        pred2so_ids = torch.argmax(att_matrx,dim=-1).t()  
        
        pred2so_ids = torch.repeat_interleave(pred2so_ids,self.topk,dim=0)

        dura_inters,dura_mask = dura_intersection_ts(traj_duras,traj_duras) 
        dura_mask[range(n_traj),range(n_traj)] = 0
        pos_pred_mask = dura_mask[pred2so_ids[:,0],pred2so_ids[:,1]] 
        if pos_pred_mask.sum() == 0:
            return None
        pos_pred_index = pos_pred_mask.nonzero(as_tuple=True)[0]

        pred2so_ids = pred2so_ids[pos_pred_index,:]  
        pred_scores =pred_scores[pos_pred_index]    
        pred_catids =pred_catids[pos_pred_index]   
        predquery_ids=predquery_ids[pos_pred_index] 

        pred2so_catids = enti_catids[pred2so_ids] 
        triplet_catids = torch.cat([pred_catids[:,None],pred2so_catids],dim=-1) 
        

        pred2so_scores = enti_scores[pred2so_ids] 
        triplet_scores = torch.cat([pred_scores[:,None],pred2so_scores],dim=-1) 

        quintuples = torch.cat([triplet_catids,pred2so_ids],dim=-1) 
        try:
            uniq_quintuples,index_map = unique_with_idx_nd(quintuples)   
        except:
            print(quintuples.shape,pos_pred_mask.shape,pos_pred_mask.sum())
            print(dura_mask.sum(),dura_mask.shape)
            print(proposal.video_name)
            assert False

        uniq_triplet_ids = [idx[triplet_scores[idx,0].argmax()] for idx in index_map] 
        uniq_triplet_ids = torch.stack(uniq_triplet_ids)                      
        uniq_scores = triplet_scores[uniq_triplet_ids,:]   
        uniq_query_ids = predquery_ids[uniq_triplet_ids]  
                               
        uniq_dura_inters = dura_inters[uniq_quintuples[:,3],uniq_quintuples[:,4],:]

        mask = uniq_quintuples[:,0] != 0
        uniq_quintuples = uniq_quintuples[mask,:]
        uniq_scores = uniq_scores[mask,:]
        uniq_query_ids = uniq_query_ids[mask]
        uniq_dura_inters = uniq_dura_inters[mask,:]


        ret = (
            uniq_quintuples, 
            uniq_scores,      
            uniq_dura_inters, 
            uniq_query_ids,    
        )

        return ret

class TripletLoss(nn.Module):

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None: 
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss
