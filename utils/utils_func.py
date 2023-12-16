import os
import tempfile
import shutil
import sys
from importlib import import_module

import numpy as np
import json
import cv2
import torch
import logging
from torchvision.ops import roi_pool as roi_pool2d


def parse_config_py(filename):

    filename = os.path.abspath(os.path.expanduser(filename))
    assert filename.endswith('.py')
    cfg_dict = {}
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(filename,
                        os.path.join(temp_config_dir, temp_config_name))
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }

        del sys.modules[temp_module_name]

        temp_config_file.close()
    return cfg_dict

def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x

def stack_with_padding(tensor_list,dim,value=0,rt_mask=False):

    shape_list = [t.shape for t in tensor_list]
    n_dim = len(shape_list[0])
    max_sp = []  
    for i in range(n_dim):
        max_sp.append(
            max([sp[i] for sp in shape_list])
        )
    aft_pad_list = []
    mask_list = []
    for tensor in tensor_list:
        sp = tensor.shape
        pad_n = [m-s for m,s in zip(max_sp,sp)]
        pad_n.reverse()
        pad_size = []
        for pn in pad_n:
            pad_size += [0,pn]
        aft_pad_list.append(
            torch.constant_pad_nd(tensor,pad_size,value=value)
        )
        if rt_mask:
            mask = torch.ones(tensor.shape,dtype=torch.bool,device=tensor.device)
            mask_list.append(
                torch.constant_pad_nd(mask,pad_size)
            )
    if rt_mask:
        return torch.stack(aft_pad_list,dim=dim),torch.stack(mask_list,dim=dim)
    else:
        return torch.stack(aft_pad_list,dim=dim)

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

def merge_consec_fg(segement_list):
    assert isinstance(segement_list,list)
    bg_ratio_th = 0.5
    num_seg = len(segement_list)
    
    after_merged_all_lvls = []
    level_1 = [(x,0) for x in segement_list]
    after_merged_all_lvls.append(level_1)

    while True:
        segs_crt_lvl = after_merged_all_lvls[-1]
        num_seg = len(segs_crt_lvl)

        segs_next_lvl = []
        for idx in range(num_seg-1):
            crt_seg,n_bg1 = segs_crt_lvl[idx]
            next_seg,n_bg2 = segs_crt_lvl[idx+1]
            
            span = next_seg[0] - crt_seg[-1] -1
            new_bgs = span if span > 0 else 0 

            num_bgs = n_bg1 + n_bg2 + new_bgs
            merged_seg = sorted(list(set(crt_seg + next_seg)))
            merged_seg = (merged_seg, num_bgs)
            if num_bgs/(len(merged_seg[0])+num_bgs) < bg_ratio_th:
                segs_next_lvl.append(merged_seg)
        if segs_next_lvl == []:
            break
        else:
            after_merged_all_lvls.append(segs_next_lvl)
    
    all_merged_segs = []
    for segs_per_lvl in after_merged_all_lvls:
        segs_per_lvl = [x[0] for x in segs_per_lvl]  
        all_merged_segs += segs_per_lvl
    return all_merged_segs


def average_to_fixed_length(visual_input,num_sample_clips):

    num_clips = visual_input.shape[0]  
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

def VidRead2ImgNpLits(video_path):
    img_list = []
    cap = cv2.VideoCapture(video_path)  
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            img_list.append(image)  
            count+=1
    
    return img_list

def VidRead2ImgTensorLits(video_path):
    img_list = []
    cap = cv2.VideoCapture(video_path)  
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    count = 0
    while success and count < n_frames:
        success, image = cap.read()
        if success:
            image = torch.from_numpy(image).permute(2,0,1).float() 
            img_list.append(image)  
            count+=1
    
    return img_list

def create_logger(filename='train.log',filemode='a',fmt='%(asctime)s - %(message)s', level=logging.DEBUG):

    logging.basicConfig(filename=filename,filemode=filemode,format=fmt, level=level)
    logger = logging.getLogger()
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)

    return logger

def traj_align_pool(traj_features,inter_dura,roi_outlen,scale):

    n_trajs,_,_ = traj_features.shape
    _,n_pos_ac,_ = inter_dura.shape

    input = traj_features.permute(0,2,1).float() 
    input = input[...,None] 

    inter_dura = inter_dura.reshape(2*n_pos_ac,-1)  
    tid = inter_dura[:,None,0]
    assert tid.max() < n_trajs
    tl = torch.constant_pad_nd(inter_dura[:,None,1],pad=(1,0))  
    br = torch.constant_pad_nd(inter_dura[:,None,2],pad=(1,0))  
    rois = torch.cat([tid,tl,br],dim=-1).float()  

    output_size = (roi_outlen,1)
    result = roi_pool2d(input,rois,output_size,spatial_scale=scale)  
    result = result.squeeze(-1).permute(0,2,1)  
    result = result.reshape(2,n_pos_ac,roi_outlen,-1)
    return result

def traj_roi_pool(traj_features,inter_dura,adj_mask,roi_outlen,scale):

    input = traj_features.permute(0,2,1) 
    input = input[...,None] 

    tl = torch.constant_pad_nd(inter_dura[:,:,None,0],pad=(1,0))  
    br = torch.constant_pad_nd(inter_dura[:,:,None,1],pad=(1,0))  
    rois = torch.cat([tl,br],dim=-1) 

    n_trajs,n_anchors,_ = rois.shape
    rois_tid = torch.tensor(list(range(n_trajs)),device=rois.device)
    rois_tid = rois_tid[:,None,None].repeat(1,n_anchors,1)  

    rois = torch.cat([rois_tid,rois],dim=-1) 
    rois = rois[adj_mask].float() 

    output_size = (roi_outlen,1)
    result = roi_pool2d(input,rois,output_size,spatial_scale=scale)  
    result = result.squeeze(-1).permute(0,2,1) 
    return result

def interpolation_single(vector_l,vector_r,left,right):
    assert left +1 < right  
    assert len(vector_l.shape) == 1
    assert vector_l.shape == vector_r.shape
    inter_len = right-left-1

    inter_vector = np.linspace(vector_l,vector_r,num=inter_len+2,axis=0)[1:-1]
    return inter_vector

def fill_zeropadding(vectors):
    mask0 = vectors == 0     
    index0 = np.where(np.all(mask0,axis=-1))[0]
    assert np.all(np.diff(index0) > 1) ,"index0={}".format(index0) 
    index_neighbor = index0 - 1
    index_neighbor[index_neighbor == -1] = 1
    vectors[index0] = vectors[index_neighbor]

def linear_interpolation(vectors,frame_ids):
   
    assert len(vectors.shape) == 2
    frame_ids = np.array(frame_ids)  
    frame_id_diff = np.diff(frame_ids)
    cut_point = np.where(frame_id_diff > 1)[0] + 1

    consec_frames = np.split(frame_ids,cut_point)
    consec_vectors = np.split(vectors,cut_point,axis=0)
    num_consecutive = len(consec_frames)

    result_vectors = []
    for i in range(1,num_consecutive,1):
        left_vector = consec_vectors[i-1][-1]
        right_vector = consec_vectors[i][0]
        fill_zeropadding(left_vector)
        fill_zeropadding(right_vector)
        left = consec_frames[i-1][-1]
        right = consec_frames[i][0]
        inter_vectors = interpolation_single(left_vector,right_vector,left,right)
        result_vectors.append(consec_vectors[i-1])
        result_vectors.append(inter_vectors)

    result_vectors.append(consec_vectors[-1])
    result_vectors = np.concatenate(result_vectors,axis=0)
    return result_vectors


def unique_with_idx(tensor):
    assert len(tensor.shape) == 1  
    unique_,counts = torch.unique(tensor,return_counts=True)
    mask = tensor[None,:] == unique_[:,None]
    index_map = mask.nonzero(as_tuple=True)[1]
    index_map = torch.split(index_map,counts.tolist())  

    return unique_,index_map

def unique_with_idx_nd(tensor):

    unique_,counts = torch.unique(tensor,return_counts=True,dim=0)

    mask = tensor[None,:,...] == unique_[:,None,...]  
    mask = mask.reshape(mask.shape[0],mask.shape[1],-1)
    mask = torch.all(mask,dim=-1) 
    index_map = mask.nonzero(as_tuple=True)[1]
    index_map = torch.split(index_map,counts.tolist()) 

    return unique_,index_map

def dura_intersection_ts(dura1,dura2,broadcast=True):

    assert isinstance(dura1,torch.Tensor) and isinstance(dura2,torch.Tensor)
    n1,n2 = dura1.shape[0],dura2.shape[0]
    mask1 = dura1[:,0] <= dura1[:,1]
    mask2 = dura2[:,0] <= dura2[:,1]
    assert mask1.sum() == n1 and mask2.sum() == n2
    
    if broadcast:
        inter_s = torch.max(dura1[:,None,0],dura2[None,:,0])
        inter_e = torch.min(dura1[:,None,1],dura2[None,:,1])
        intersection = torch.stack([inter_s,inter_e],dim=-1)
        mask = intersection[:,:,0] <= intersection[:,:,1]  
    else:
        assert n1 == n2
        inter_s = torch.max(dura1[:,0],dura2[:,0])
        inter_e = torch.min(dura1[:,1],dura2[:,1])
        intersection = torch.stack([inter_s,inter_e],dim=-1) 
        mask = intersection[:,0] <= intersection[:,1]   

    return intersection,mask

def tIoU(duras1,duras2,broadcast=True):

    if broadcast:
        mask = (duras1[:,None,1] >= duras2[None,:,0]) * (duras2[None,:,1] >= duras1[:,None,0])  
        tiou = (torch.min(duras1[:,None,1],duras2[None,:,1]) - torch.max(duras1[:,None,0],duras2[None,:,0])) \
            / (torch.max(duras1[:,None,1],duras2[None,:,1]) - torch.min(duras1[:,None,0],duras2[None,:,0]))
    else:
        assert duras1.shape == duras2.shape
        mask = (duras1[:,1] >= duras2[:,0]) * (duras2[:,1] >= duras1[:,0])  
        tiou = (torch.min(duras1[:,1],duras2[:,1]) - torch.max(duras1[:,0],duras2[:,0])) \
            / (torch.max(duras1[:,1],duras2[:,1]) - torch.min(duras1[:,0],duras2[:,0]))

    tiou[torch.logical_not(mask)] = 0

    return tiou 


def generalized_tIoU(duras1,duras2,broadcast=True):

    if broadcast:
        g_tiou = (torch.min(duras1[:,None,1],duras2[None,:,1]) - torch.max(duras1[:,None,0],duras2[None,:,0])) \
            / (torch.max(duras1[:,None,1],duras2[None,:,1]) - torch.min(duras1[:,None,0],duras2[None,:,0]))
    else:
        assert duras1.shape == duras2.shape
        g_tiou = (torch.min(duras1[:,1],duras2[:,1]) - torch.max(duras1[:,0],duras2[:,0])) \
            / (torch.max(duras1[:,1],duras2[:,1]) - torch.min(duras1[:,0],duras2[:,0]))


    return g_tiou  

def vIoU_ts_rel(traj_1,traj_2,dura_1,dura_2):

    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert isinstance(dura_1,torch.Tensor) and isinstance(dura_2,torch.Tensor)
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    traj_1 = traj_1[dura_1[0]:dura_1[1]+1,:]
    traj_2 = traj_2[dura_2[0]:dura_2[1]+1,:]
    assert traj_1.shape == traj_2.shape 

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def vIoU_ts(traj_1,traj_2,dura_1,dura_2):

    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert isinstance(dura_1,torch.Tensor) and isinstance(dura_2,torch.Tensor)
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    traj_1 = traj_1[dura_1[0]:dura_1[1]+1,:]
    traj_2 = traj_2[dura_2[0]:dura_2[1]+1,:]
    assert traj_1.shape == traj_2.shape  

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def vIoU_aligned(traj_1,traj_2):

    assert isinstance(traj_1,torch.Tensor) and isinstance(traj_2,torch.Tensor)
    assert traj_1.shape == traj_2.shape
    
    traj_1 = traj_1.float()
    traj_2 = traj_2.float()
    TO_REMOVE = 1
    area_1 = (traj_1[:, 2] - traj_1[:, 0] + TO_REMOVE) * (traj_1[:, 3] - traj_1[:, 1] + TO_REMOVE)
    area_2 = (traj_2[:, 2] - traj_2[:, 0] + TO_REMOVE) * (traj_2[:, 3] - traj_2[:, 1] + TO_REMOVE)

    lt = torch.max(traj_1[:,:2],traj_2[:,:2])
    rb = torch.min(traj_1[:,2:],traj_2[:,2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0.0)
    inter_area = (wh[:,0] * wh[:,1]).sum()
    
    return inter_area / (area_1.sum() + area_2.sum() - inter_area)


def dura_intersection(dura1,dura2):
    s1,e1 = dura1
    assert s1 < e1 ,"dura1={},dura2={}".format(dura1,dura2)
    s2,e2 = dura2
    assert s2 < e2 ,"dura1={},dura2={}".format(dura1,dura2)
    if e1 <= s2 or e2 <= s1:        

        return None
    
    inter_s = max(s1,s2)
    inter_e = min(e1,e2)

    return (inter_s, inter_e)

def traj_cutoff_close(ori_traj,ori_dura,dura,debug_info=None):

    assert len(ori_traj) == ori_dura[1] - ori_dura[0] + 1,"len(traj)={}!=end_fid-start_fid={},{}".format(len(ori_traj),ori_dura[1] - ori_dura[0],debug_info)
    s_o, e_o = ori_dura
    ss, ee = dura
    assert s_o <= ss and ee <= e_o,"ori_dura={},dura={},{}".format(ori_dura,dura,debug_info)

    index_s = ss - s_o
    index_e = index_s + (ee - ss)  
    return ori_traj[index_s:index_e]

def traj_cutoff(ori_traj,ori_dura,dura,debug_info=None):

    assert len(ori_traj) == ori_dura[1] - ori_dura[0],"len(traj)={}!=end_fid-start_fid={},{}".format(len(ori_traj),ori_dura[1] - ori_dura[0],debug_info)
    s_o, e_o = ori_dura
    ss, ee = dura
    assert s_o <= ss and ee <= e_o,"ori_dura={},dura={},{}".format(ori_dura,dura,debug_info)

    index_s = ss - s_o
    index_e = len(ori_traj) - (e_o - ee)
    return ori_traj[index_s:index_e]

def vIoU(traj_1, duration_1, traj_2, duration_2):

    assert type(traj_1) == type(traj_2), "{}, {}".format(type(traj_1),type(traj_2))
    if isinstance(traj_1,torch.Tensor):
        traj_1 = traj_1.float()
        traj_2 = traj_2.float()
    elif isinstance(traj_1,np.ndarray):
        traj_1 = traj_1.astype(np.float32)
        traj_2 = traj_2.astype(np.float32)
    else:
        assert isinstance(traj_1,list)

    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.0
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)

def merge_duration_list(duration_list):

    duration_list = duration_list.copy()
    duration_list = sorted(duration_list,key=lambda d: d[0])  
    merged_durations = []
    head_dura = duration_list.pop(0)
    merged_durations.append(head_dura)

    while duration_list != []:
        former_dura = merged_durations[-1]
        former_start,former_end = former_dura

        cur_dura = duration_list.pop(0)
        cur_start,cur_end = cur_dura
        if cur_start <= former_end:
            merged_durations.pop(-1)
            merged_dura = (former_start,cur_end)
            merged_durations.append(merged_dura)
        else:
            merged_durations.append(cur_dura)

    return merged_durations

def is_overlap_old(dura1,dura2):
    dura_list = [dura1,dura2]
    dura_list = dura_list.copy()
    dura_list = sorted(dura_list,key=lambda d: d[0]) 
    d1_start,d1_end = dura_list[0]
    assert d1_start <= d1_end
    d2_start,d2_end = dura_list[1]
    assert d2_start <= d2_end

    if d2_start < d1_end:
        return True
    else:
        return False

def is_overlap(dura1,dura2):
    s1,e1 = dura1
    assert s1 < e1 
    s2,e2 = dura2
    assert s2 < e2

    if e1 <= s2 or e2 <= s1:

        return False
    else:
        return True

