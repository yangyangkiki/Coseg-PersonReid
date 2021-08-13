#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F

def LocationLoss(batch_coordinate, lloss_parts,cosine_criterion):
    B, C, _ = batch_coordinate.shape
    loss = 0
    n = 0
    for i in range(batch_coordinate.size(0)):  # batch
        anchor = batch_coordinate[i][0]
        # vectors = torch.zeros([lloss_parts-1, 2]).cuda()  # 不能这样新建
        vectors = []
        for j in range(1, lloss_parts):
            n = n + 1
            positive = batch_coordinate[i][j]
            vector = positive - anchor
            vectors.append(torch.stack((vector[0], vector[1]), dim=0).unsqueeze(0))
        # v2
        comb = combinations(vectors, 2)
        for k in list(comb):
            loss = loss + (1 - cosine_criterion(k[0][0], k[1][0]).abs())  # print(k)

    loss = loss / n
    return loss

# # calculate per img cosine loss --v1
# vectors = torch.cat(vectors, dim=0)
# a = vectors[0]
# b = vectors[1]
# c = vectors[2]
# loss = loss + (1-cosine_criterion(a, b).abs())
# loss = loss + (1-cosine_criterion(a, c).abs())


def get_coordinate_tensors(x_max, y_max):
    # x_map = np.tile(np.arange(x_max), (y_max, 1)) / x_max * 2 - 1.0  # h center :
    # y_map = np.tile(np.arange(y_max), (x_max, 1)).T / y_max * 2 - 1.0  # w center : 0-7 chongfu 24 bian

    # # zhao yang
    # x_map = np.transpose(x_map)
    # y_map = np.transpose(y_map)
    x_map = np.tile(np.arange(x_max), (y_max, 1)).T / x_max * 2 - 1.0  # h center
    y_map = np.tile(np.arange(y_max), (x_max, 1)) / y_max * 2 - 1.0  # w center
    # # zhao yang

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor

def get_variance(part_map, x_c, y_c):

    h, w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    v_x_map = (x_map - x_c) * (x_map - x_c)
    v_y_map = (y_map - y_c) * (y_map - y_c)

    v_x = (part_map * v_x_map).sum()
    v_y = (part_map * v_y_map).sum()
    return v_x, v_y

def get_center(part_map, self_referenced=False):

    h, w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C, H, W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c, :, :] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map / k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)  # self_ref_coord
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax):
    B, C, H, W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

def concentrationloss(pred_softmax):

    pred_softmax = pred_softmax[:, 1:, :, :]  # last is background

    B, C, H, W = pred_softmax.shape

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(pred_softmax)
    for b in range(B):
        centers = centers_all[b]
        for c in range(C):
            # normalize part map as spatial pdf
            part_map = pred_softmax[b, c, :, :] + epsilon  # prevent gradient explosion
            k = part_map.sum()
            part_map_pdf = part_map / k
            x_c, y_c = centers[c]
            v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
            loss_per_part = (v_x + v_y)
            loss = loss_per_part + loss
    loss = loss / B
    return loss / B

def heightloss(pred_softmax, margin):

    pred_softmax = pred_softmax[:, 1:, :, :]  # last is background
    B, C, H, W = pred_softmax.shape

    ys = torch.arange(0, H, dtype=torch.float32, device=pred_softmax.device)
    xs = torch.arange(0, W, dtype=torch.float32, device=pred_softmax.device)

    coords = torch.stack([xs[None,:].repeat(H,1),ys[:,None].repeat(1,W)],dim=2)
    coords = torch.unsqueeze(torch.unsqueeze(coords,0),0)

    pred_softmax = torch.unsqueeze(pred_softmax, 4)

    xx = pred_softmax * coords
    coords_centers = torch.mean(torch.mean(xx, 2), 2)

    coords_centers_x = coords_centers[:, :, 0]
    coords_centers_y = coords_centers[:, :, 1]

    # calculate loss
    y = coords_centers_y[:,0].new().resize_as_(coords_centers_y[:,0]).fill_(1)
    max_cal = coords_centers_y.size(1)
    loss_all = 0
    for i in range(max_cal):
        if i+1 != max_cal:
            dist_an = coords_centers_y[:, i+1]
            dist_ap = coords_centers_y[:, i]

            if margin > 0:
                loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
            else:
                loss = F.soft_margin_loss(dist_an - dist_ap, y)

                if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

            loss_all = loss_all + loss

    return loss_all/(max_cal-1)
