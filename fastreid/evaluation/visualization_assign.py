#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import colorsys
import matplotlib.image as mpimg
from PIL import Image
import argparse
import os
import sys
from os import mkdir
import numpy as np
import torch
from torch.backends import cudnn
from matplotlib import pyplot as plt
sys.path.append('.')
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import shutil

def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

def generate_colors(num_colors):
    """
    Generate distinct value by sampling on hls domain.

    Parameters
    ----------
    num_colors: int
        Number of colors to generate.

    Returns
    ----------
    colors_np: np.array, [num_colors, 3]
        Numpy array with rows representing the colors.

    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors)*255.

    return colors_np

def plot_assignment(root, assign_hard, num_parts,img_name,size=None):
    """
    Blend the original image and the colored assignment maps.

    Parameters
    ----------
    root: str
        Root path for saving visualization results.
    assign_hard: np.array, [H, W]
        Hard assignment map (int) denoting the deterministic assignment of each pixel. Generated via argmax.
    num_parts: int, number of object parts.

    Returns
    ----------
    Save the result to root/assignment.png.

    """
    size = size
    # generate the numpy array for colors
    colors = generate_colors(num_parts)

    # coefficient for blending
    coeff = 0.4

    # load the input as RGB image, convert into numpy array
    # input = Image.open(os.path.join(root, 'input.png')).convert('RGB')
    input = Image.open(os.path.join(root, img_name)).convert('RGB')
    input = input.resize((size[1], size[0]), Image.ANTIALIAS)
    input_np = np.array(input).astype(float)

    # blending by each pixel
    for i in range(assign_hard.shape[0]):
        for j in range(assign_hard.shape[1]):
            assign_ij = assign_hard[i][j]
            input_np[i, j] = (1-coeff) * input_np[i, j] + coeff * colors[assign_ij]

    # save the resulting image
    im = Image.fromarray(np.uint8(input_np))
    # im.save(os.path.join(root, 'assignment.png'))
    im.save(os.path.join(root, img_name))

# zhao yang  # 画图
def visualization_assignment(cfg,model, val_loader):
    model.eval()
    dataset_name = cfg.DATASETS.NAMES[0]
    # create a dataloader iter instance
    test_loader_iter = iter(val_loader)  # 如果想要看trainset的图片的话就要去dataset里面改 self.train = self.train self.query = self.train self.train[:2]
    ccl = 0
    print(cfg.INTERPRATABLE.NPARTS, 'parts')
    for index in range(len(test_loader_iter)):
        ccl=ccl+1
        if ccl == cfg.INTERPRATABLE.VISUALIZE_NUM:
            break
        with torch.no_grad():
            # the visualization code
            # current_id = 0

            # data, pids, camids, pic_dir = next(test_loader_iter)
            data = next(test_loader_iter)
            input = data['images'].cuda()
            pic_dir =data['img_paths']

            assign = model(data)
            # _,_,att_list, assign,_,_ = model(input)

            # define root for saving results and make directories correspondingly
            root = os.path.join('./visualization', dataset_name)  # str(current_id)root = os.path.join('./visualization', dataset_name, pic_dir[0].split('/')[-1])
            os.makedirs(root, exist_ok=True)

            # # denormalize the image and save the input
            # # save_input = transforms.Normalize(mean=(0, 0, 0),std=(1/0.229, 1/0.224, 1/0.225))(input.data[0].cpu())
            # # save_input = transforms.Normalize(mean=(-0.485, -0.456, -0.406),std=(1, 1, 1))(save_input)
            #
            # save_input = transforms.Normalize(mean=(0, 0, 0), std=(1 / 0.229* 255, 1 / 0.224* 255, 1 / 0.225* 255))(
            #     input.data[0].cpu())
            # save_input = transforms.Normalize(mean=(-0.485 * 255, -0.456 * 255, -0.406 * 255), std=(1, 1, 1))(save_input)
            #
            # save_input = Image.open(pic_dir[0])
            # save_input = np.array(save_input)
            # save_input = torch.tensor(save_input).permute(2,0,1)
            #
            # save_input = torch.nn.functional.interpolate(save_input.unsqueeze(0), size=cfg.INPUT.SIZE_TEST, mode='bilinear', align_corners=False).squeeze(0)
            # img = torchvision.transforms.ToPILImage()(save_input)
            # # img.save(os.path.join(root, 'input.png'))  #pic_dir[0].split('/')[-1]
            shutil.copy(pic_dir[0], os.path.join(root, pic_dir[0].split('/')[-1]))
            # img.save(os.path.join(root, pic_dir[0].split('/')[-1]))

            # upsample the assignment and transform the attention correspondingly
            assign_reshaped = torch.nn.functional.interpolate(assign.data.cpu(), size=cfg.INPUT.SIZE_TEST, mode='bilinear',
                                                              align_corners=False)

            # # visualize the attention
            # for k in range(1):
            #     # attention vector for kth attribute
            #     att = att_list[k].view(
            #         1, cfg.INTERPRATABLE.NPARTS, 1, 1).data.cpu()
            #
            #     # multiply the assignment with the attention vector
            #     assign_att = assign_reshaped * att
            #
            #     # sum along the part dimension to calculate the spatial attention map   # ?
            #     attmap_hw = torch.sum(assign_att, dim=1).squeeze(0).numpy()
            #
            #     # normalize the attention map and merge it onto the input
            #     img = cv2.imread(os.path.join(root, pic_dir[0].split('/')[-1]))
            #     mask = attmap_hw / attmap_hw.max()  # ? 0-1
            #     img_float = img.astype(float) / 255.
            #     ddr = os.path.join(root, 'attentions')
            #     if ddr and not os.path.exists(ddr):
            #         mkdir(ddr)
            #     show_att_on_image(img_float, mask, os.path.join(ddr, pic_dir[0].split('/')[-1]))  # , 'attentions'

            # color_att = mpimg.imread(os.path.join(root, 'attentions' + '.png'))
            # axarr_assign_att[j, col_id].imshow(color_att)
            # axarr_assign_att[j, col_id].axis('off')

            # generate the one-channel hard assignment via argmax
            _, assign = torch.max(assign_reshaped, 1)

            # colorize and save the assignment
            plot_assignment(root, assign.squeeze(0).numpy(), cfg.INTERPRATABLE.NPARTS, pic_dir[0].split('/')[-1], size=cfg.INPUT.SIZE_TEST)
            # 画每个assign:
            # collect the assignment for the final image array
            color_assignment_name = os.path.join(root, pic_dir[0].split('/')[-1])
            color_assignment = mpimg.imread(color_assignment_name)

            os.makedirs(os.path.join(root, pic_dir[0].split('/')[-1].split('.')[0]), exist_ok=True)

            # plot the assignment for each dictionary vector
            for i in range(cfg.INTERPRATABLE.NPARTS):
                img = torch.nn.functional.interpolate(assign_reshaped.data[:, i].cpu().unsqueeze(0),
                                                      size=cfg.INPUT.SIZE_TEST, mode='bilinear', align_corners=False)
                img = torchvision.transforms.ToPILImage()(img.squeeze(0))
                img.save(os.path.join(root, pic_dir[0].split('/')[-1].split('.')[0], 'part_' + str(i) + '.png'))

            # save the array version
            # os.makedirs('./visualization/collected', exist_ok=True)
            # f_assign.savefig(os.path.join('./visualization/collected', 'assign.png'))
            # f_assign_att.savefig(os.path.join('./visualization/collected', 'attention.png'))


    print('Visualization finished!')

# @contextmanager
