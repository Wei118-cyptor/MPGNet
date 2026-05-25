# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import random

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.roi_heads.box_head import build_box_head

from .mpgnet_fast_rcnn import mpgnetFastRCNNOutputLayers
from ..debug import debug_second_stage

@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.add_image_box = cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX
        self.add_feature_to_prop = cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.box_predictor = mpgnetFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        self.save_debug = cfg.SAVE_DEBUG
        self.save_debug_path = cfg.SAVE_DEBUG_PATH

        if self.save_debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.bgr = (cfg.INPUT.FORMAT == 'BGR')

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None,
                ann_type='box', classifier_info=(None, None, None),td=None):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if not self.save_debug:
            del images

        if self.training:
            if ann_type in ['box']:
                labeled_proposals = self.label_and_sample_proposals(proposals, targets)
                # 原始方法：简单选择 Top-K 高置信度 proposals
                # top_proposals = self.get_top_proposals(proposals)

                # 新方法：基于类别和 IoU 选择 Top-K proposals
                top_proposals = self.label_and_sample_top_proposals(proposals, targets, td)
                 
                proposal_boxes_labeled = [x.proposal_boxes for x in labeled_proposals]
                proposal_boxes_top = [x.proposal_boxes for x in top_proposals]

                box_features_labeled = self._shared_roi_transform(
                    [features[f] for f in self.in_features], proposal_boxes_labeled
                        )
                box_features_top = self._shared_roi_transform(
                    [features[f] for f in self.in_features], proposal_boxes_top
                        )

                predictions_labeled = self.box_predictor(
                    box_features_labeled.mean(dim=[2, 3]),
                    ann_type=ann_type, classifier_info=classifier_info)
                predictions_top = self.box_predictor(
                    box_features_top.mean(dim=[2, 3]),
                    ann_type=ann_type, classifier_info=classifier_info)

                # 如果需要添加feature到proposals
                if self.add_feature_to_prop:
                # 处理labeled_proposals
                    box_features_mean = box_features_labeled.mean(dim=[2, 3])
                    box_features_cls = self.box_predictor.cls_score.linear(box_features_mean)
                    feats_per_image = box_features_cls.split(
                        [len(p) for p in labeled_proposals], dim=0)
                    for feat, p in zip(feats_per_image, labeled_proposals):
                        p.feat = feat
                            
                # 处理top_proposals
                    box_features_mean = box_features_top.mean(dim=[2, 3])
                    box_features_cls = self.box_predictor.cls_score.linear(box_features_mean)
                    feats_per_image = box_features_cls.split(
                        [len(p) for p in top_proposals], dim=0)
                    for feat, p in zip(feats_per_image, top_proposals):
                        p.feat = feat  
            else:
                proposals = self.get_top_proposals(proposals)
                proposal_boxes = [x.proposal_boxes for x in proposals]
                box_features = self._shared_roi_transform(
                    [features[f] for f in self.in_features], proposal_boxes
                )

                predictions = self.box_predictor(
                    box_features.mean(dim=[2, 3]),
                    ann_type=ann_type, classifier_info=classifier_info)

                if self.add_feature_to_prop:
                    box_features = box_features.mean(dim=[2, 3])
                    box_features = self.box_predictor.cls_score.linear(box_features)
                    feats_per_image = box_features.split(
                        [len(p) for p in proposals], dim=0)
                    for feat, p in zip(feats_per_image, proposals):
                        p.feat = feat
        else:
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self._shared_roi_transform(
                [features[f] for f in self.in_features], proposal_boxes
            )
            predictions = self.box_predictor(
                box_features.mean(dim=[2, 3]),
                ann_type=ann_type, classifier_info=classifier_info)
            
            if self.add_feature_to_prop:
                box_features = box_features.mean(dim=[2, 3])
                box_features = self.box_predictor.cls_score.linear(box_features)
                feats_per_image = box_features.split(
                    [len(p) for p in proposals], dim=0)
                for feat, p in zip(feats_per_image, proposals):
                    p.feat = feat

        if self.training:
            del features
            if (ann_type == 'box'):
                labeled_losses = self.box_predictor.losses(
                        (predictions_labeled[0], predictions_labeled[1]), labeled_proposals,
                        classifier_info=classifier_info)
        
                image_labels = [x.gt_classes.unique().tolist() for x in targets]
                top_losses = self.box_predictor.image_label_losses(
                    predictions_top, top_proposals, image_labels,
                    classifier_info=classifier_info,td=td)
                    
                losses = labeled_losses.copy()
                if 'image_loss' in top_losses:
                    losses['image_loss'] = top_losses['image_loss']
                    
                if self.with_image_labels:
                    if 'image_loss' not in losses:
                        losses['image_loss'] = predictions_labeled[0].new_zeros([1])[0]
                return labeled_proposals, losses
            else:
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, image_labels,
                    classifier_info=classifier_info, ann_type=ann_type)
                return proposals, losses
        else:
            # 在非训练模式下，直接使用输入的 proposals         
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
        
    @torch.no_grad()
    def label_and_sample_top_proposals(self, proposals, targets, td=None):
        """
        新的采样策略：
        1. 像 label_and_sample_proposals 一样进行 IoU 匹配和类别分配
        2. 但是采样时，根据类别筛选正样本，并按 IoU 选择 Top-K
        3. 如果提供了 td (true_ids)，则只选择与目标类别ID匹配的正样本
        
        Args:
            proposals: List[Instances], RPN 生成的 proposals
            targets: List[Instances], ground truth
            td: 目标类别ID张量，用于筛选匹配的正样本
            
        Returns:
            List[Instances]: 每张图像采样 ws_num_props 个 proposals
        """
        proposals_with_gt = []

        for img_idx, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            has_gt = len(targets_per_image) > 0
            
            if not has_gt:
                # 如果没有GT，直接取前 ws_num_props 个
                proposals_per_image.proposal_boxes.clip(proposals_per_image.image_size)
                selected_proposals = proposals_per_image[:self.ws_num_props]
                proposals_with_gt.append(selected_proposals)
                continue
            
            # 计算 IoU 矩阵
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs: 每个proposal匹配的GT索引
            # matched_labels: 每个proposal的匹配标签(1=正样本, 0=负样本, -1=忽略)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            
            # 为每个proposal分配GT类别
            gt_classes = targets_per_image.gt_classes[matched_idxs]
            # 将不匹配的proposals标记为背景
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
            
            # 找出所有正样本的索引
            positive_mask = (matched_labels == 1)
            
            if td is not None and positive_mask.sum() > 0:
                # 如果提供了 td，则筛选与目标类别ID匹配的正样本
                target_class_id = int(td[img_idx].item()) if hasattr(td[img_idx], 'item') else int(td[img_idx])
                # 只保留类别匹配的正样本
                class_matched_mask = (gt_classes == target_class_id) & positive_mask
                
                if class_matched_mask.sum() > 0:
                    # 获取匹配的proposal索引
                    matched_indices = torch.nonzero(class_matched_mask).squeeze(1)
                    # 获取这些proposals对应的最大IoU值
                    max_ious = match_quality_matrix[:, matched_indices].max(dim=0)[0]
                    # 按IoU降序排序
                    sorted_indices = torch.argsort(max_ious, descending=True)
                    # 选择Top-K
                    top_k = min(self.ws_num_props, len(sorted_indices))
                    selected_local_indices = sorted_indices[:top_k]
                    sampled_idxs = matched_indices[selected_local_indices]
                else:
                    # 如果没有类别匹配的正样本，回退到选择所有正样本
                    positive_indices = torch.nonzero(positive_mask).squeeze(1)
                    if len(positive_indices) > 0:
                        max_ious = match_quality_matrix[:, positive_indices].max(dim=0)[0]
                        sorted_indices = torch.argsort(max_ious, descending=True)
                        top_k = min(self.ws_num_props, len(sorted_indices))
                        sampled_idxs = positive_indices[sorted_indices[:top_k]]
                    else:
                        # 如果连正样本都没有，选择前 ws_num_props 个
                        sampled_idxs = torch.arange(min(self.ws_num_props, len(proposals_per_image)), 
                                                   device=proposals_per_image.proposal_boxes.tensor.device)
            else:
                # 如果没有提供 td，按原有逻辑：选择所有正样本，按IoU排序
                if positive_mask.sum() > 0:
                    positive_indices = torch.nonzero(positive_mask).squeeze(1)
                    max_ious = match_quality_matrix[:, positive_indices].max(dim=0)[0]
                    sorted_indices = torch.argsort(max_ious, descending=True)
                    top_k = min(self.ws_num_props, len(sorted_indices))
                    sampled_idxs = positive_indices[sorted_indices[:top_k]]
                else:
                    # 如果没有正样本，选择前 ws_num_props 个
                    sampled_idxs = torch.arange(min(self.ws_num_props, len(proposals_per_image)), 
                                               device=proposals_per_image.proposal_boxes.tensor.device)
            
            # 提取选中的proposals
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes[sampled_idxs]
            
            # 设置GT属性
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals

    def _add_image_box(self, p, use_score=False):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        if self.image_box_size < 1.0:
            f = self.image_box_size
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [w * (1. - f) / 2.,
                     h * (1. - f) / 2.,
                     w * (1. - (1. - f) / 2.),
                     h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        else:
            image_box.proposal_boxes = Boxes(
                p.proposal_boxes.tensor.new_tensor(
                    [0, 0, w, h]).view(n, 4))
        if use_score:
            image_box.scores = \
                p.objectness_logits.new_ones(n)
            image_box.pred_classes = \
                p.objectness_logits.new_zeros(n, dtype=torch.long)
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n)
        else:
            image_box.objectness_logits = \
                p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])
