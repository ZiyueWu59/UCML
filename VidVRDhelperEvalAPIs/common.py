import numpy as np


def voc_ap(rec, prec, use_07_metric=False):

    if use_07_metric:
        
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def iou(bbox_1, bbox_2):

    w_1 = bbox_1[2] - bbox_1[0] + 1
    h_1 = bbox_1[3] - bbox_1[1] + 1
    w_2 = bbox_2[2] - bbox_2[0] + 1
    h_2 = bbox_2[3] - bbox_2[1] + 1
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    overlap_bbox = (max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
                    min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3]))
    overlap_w = max(0, (overlap_bbox[2] - overlap_bbox[0] + 1))
    overlap_h = max(0, (overlap_bbox[3] - overlap_bbox[1] + 1))

    overlap_area = overlap_w * overlap_h
    union_area = area_1 + area_2 - overlap_area
    IoU = overlap_area * 1.0 / union_area
    return IoU


def viou(traj_1, duration_1, traj_2, duration_2):

    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.
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