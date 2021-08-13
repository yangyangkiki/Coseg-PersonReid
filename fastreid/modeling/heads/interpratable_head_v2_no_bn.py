# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
from .grouping import GroupingUnit

# Bottleneck of standard ResNet50/101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Basicneck of standard ResNet18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@REID_HEADS_REGISTRY.register()
class InterpratableHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()  # this one
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*bottleneck)

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, feat_dim, num_classes)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        # zhaoyang - interpratable
        self.visualization = cfg.INTERPRATABLE.VISUALIZATION
        # the grouping module
        self.num_parts = cfg.INTERPRATABLE.NPARTS
        self.part_dim = cfg.INTERPRATABLE.PART_DIM
        self.grouping = GroupingUnit(512 * 4, self.num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)

        # post-processing bottleneck block for the region features
        self.post_block = nn.Sequential(
            Bottleneck1x1(1024 * 2, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024 * 2, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048))),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
        )

        self.decrease_dim_block = nn.Sequential(  # pcb 降维度
            nn.Conv2d(2048, self.part_dim, 1, 1, bias=False),  # 降低维度
            nn.BatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True))

        # the final batchnorm
        # self.groupingbn = nn.BatchNorm2d(512 * 4)
        bottleneck_part = nn.ModuleList()
        feat_dim_part = self.part_dim * self.num_parts
        if with_bnneck:
            bottleneck_part.append(get_norm(norm_type, feat_dim_part, bias_freeze=True))

        self.bottleneck_part = nn.Sequential(*bottleneck_part)

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':
            self.classifier_part = nn.Linear(feat_dim_part, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier_part = ArcSoftmax(cfg, feat_dim_part, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier_part = CircleSoftmax(cfg, feat_dim_part, num_classes)
        elif cls_type == 'amSoftmax':
            self.classifier_part = AMSoftmax(cfg, feat_dim_part, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        # initialize convolutional layers with kaiming_normal_, BatchNorm with weight 1, bias 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the last bn in residual blocks with weight zero
        for m in self.modules():
            if isinstance(m, Bottleneck) or isinstance(m, Bottleneck1x1):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        # global f
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # part f
        self.bottleneck_part.apply(weights_init_kaiming)
        self.classifier_part.apply(weights_init_classifier)

    def forward(self, features, targets=None):

        if self._cfg.INTERPRATABLE.ONLY_INTER_HEAD:
            """
            only inter head.
            """
            global_feat = self.pool_layer(features)
            bn_feat = self.bottleneck(global_feat)
            bn_feat = bn_feat[..., 0, 0]

            # inter
            # grouping module upon the feature maps outputed by the backbone
            region_feature, assign = self.grouping(features)
            region_feature = region_feature.contiguous().unsqueeze(3)

            # non-linear layers over the region features -- GNN
            region_feature = self.post_block(region_feature)
            region_feature = region_feature.contiguous().squeeze(3)

            # average all region features into one vector based on the attention
            region_feature = F.avg_pool1d(region_feature, self.num_parts)  # * self.n_parts  # why * n_part

            # linear classifier
            part_feat = region_feature.contiguous().unsqueeze(3)  # .view(region_feature.size(0), -1)
            bn_part_feat = self.bottleneck_part(part_feat)  # BN
            bn_part_feat = bn_part_feat[..., 0, 0]
            # inter

            # Visualzation
            if self.visualization:
                return assign

            # Evaluation
            # fmt: off
            if not self.training: return bn_part_feat  # bn_feat
            # fmt: on

            # Training
            if self.classifier.__class__.__name__ == 'Linear':
                cls_outputs = self.classifier(bn_feat)
                pred_class_logits = F.linear(bn_feat, self.classifier.weight)
                # part
                cls_outputs_part = self.classifier_part(bn_part_feat)
                pred_class_logits_part = F.linear(bn_part_feat, self.classifier_part.weight)
            else:
                cls_outputs = self.classifier(bn_feat, targets)
                pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                                 F.normalize(self.classifier.weight))
                # part
                cls_outputs_part = self.classifier_part(bn_part_feat, targets)
                pred_class_logits_part = self.classifier_part.s * F.linear(F.normalize(bn_part_feat),
                                                                           F.normalize(self.classifier_part.weight))

            # fmt: off
            if self.neck_feat == "before":
                feat = global_feat[..., 0, 0]
                feat_part = part_feat[..., 0, 0]
            elif self.neck_feat == "after":
                feat = bn_feat
                feat_part = bn_part_feat
            else:
                raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
            # fmt: on

            return {
                # "cls_outputs": cls_outputs,
                # "pred_class_logits": pred_class_logits,
                # "features": feat,
                "cls_outputs_part": cls_outputs_part,
                "pred_class_logits_part": pred_class_logits_part,
                "features_part": feat_part,
                "soft_assign": assign
            }
        else:

            """
            See :class:`ReIDHeads.forward`.
            """
            global_feat = self.pool_layer(features)
            bn_feat = self.bottleneck(global_feat)
            bn_feat = bn_feat[..., 0, 0]

            # inter
            # grouping module upon the feature maps outputed by the backbone
            region_feature, assign = self.grouping(features)
            region_feature = region_feature.contiguous().unsqueeze(3)

            # non-linear layers over the region features -- GNN
            region_feature = self.post_block(region_feature)

            # region_feature = self.decrease_dim_block(region_feature)
            #
            # region_feature = region_feature.contiguous().view(region_feature.size(0), -1)
            #
            # # linear classifier
            # region_feature = region_feature.contiguous().unsqueeze(2)  # .view(region_feature.size(0), -1)
            # part_feat = region_feature.contiguous().unsqueeze(3)
            # bn_part_feat = self.bottleneck_part(part_feat)  # BN
            # bn_part_feat = bn_part_feat[..., 0, 0]
            # # inter

            part_feat = self.decrease_dim_block(region_feature)
            bn_part_feat = part_feat.contiguous().view(region_feature.size(0), -1)


            # Visualzation
            if self.visualization:
                return assign

            # Evaluation
            # fmt: off
            if not self.training: return torch.cat((bn_feat, bn_part_feat), 1)  #bn_feat
            # fmt: on

            # Training
            if self.classifier.__class__.__name__ == 'Linear':
                cls_outputs = self.classifier(bn_feat)
                pred_class_logits = F.linear(bn_feat, self.classifier.weight)
                # part
                cls_outputs_part = self.classifier_part(bn_part_feat)
                pred_class_logits_part = F.linear(bn_part_feat, self.classifier_part.weight)
            else:
                cls_outputs = self.classifier(bn_feat, targets)
                pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                                 F.normalize(self.classifier.weight))
                # part
                cls_outputs_part = self.classifier_part(bn_part_feat, targets)
                pred_class_logits_part = self.classifier_part.s * F.linear(F.normalize(bn_part_feat),
                                                                 F.normalize(self.classifier_part.weight))

            # fmt: off
            if self.neck_feat == "before":
                feat = global_feat[..., 0, 0]
                feat_part = part_feat[..., 0, 0]
            elif self.neck_feat == "after":
                feat = bn_feat
                feat_part = bn_part_feat
            else:
                raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
            # fmt: on

            return {
                "cls_outputs": cls_outputs,
                "pred_class_logits": pred_class_logits,
                "features": feat,
                "cls_outputs_part": cls_outputs_part,
                "pred_class_logits_part": pred_class_logits_part,
                "features_part": feat_part,
                "soft_assign": assign
            }
