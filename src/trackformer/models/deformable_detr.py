# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from .detr import DETR, PostProcess, SetCriterion

from target_bbox_transforms import bbox_xywh_to_xyxy

from target_transforms import inverse_min_max_scaling
from multicam_wildtrack_torch_3D_to_2D import load_spec_extrinsics
from multicam_wildtrack_torch_3D_to_2D import load_spec_intrinsics
from multicam_wildtrack_torch_3D_to_2D import transform_3D_cylinder_to_2D_COCO_bbox_params

from wildtrack_globals import N_CAMS

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(DETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 multi_frame_attention=False, multi_frame_encoding=False, merge_frame_features=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss)

        self.merge_frame_features = merge_frame_features
        self.multi_frame_attention = multi_frame_attention
        self.multi_frame_encoding = multi_frame_encoding
        self.overflow_boxes = overflow_boxes
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(backbone.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        #-----------------------------------------------------------------------
        # TOBIAS: init camera encoder

        # The camera-specific embedding vectors are in (cam, :, 0, 0)
        # I choose to intialize from N(0,1) distribution, similar to how
        # input images are normalized
        self.cam_embedder = [nn.Parameter(torch.randn(N_CAMS, emb_dim, 1, 1)) for emb_dim in num_channels]
        #-----------------------------------------------------------------------

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for
        # region proposal generation
        num_pred = transformer.decoder.num_layers
        if two_stage:
            num_pred += 1

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.merge_frame_features:
            self.merge_features = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = _get_clones(self.merge_features, num_feature_levels)

    # def fpn_channels(self):
    #     """ Returns FPN channels. """
    #     num_backbone_outs = len(self.backbone.strides)
    #     return [self.hidden_dim, ] * num_backbone_outs

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        #-----------------------------------------------------------------------
        # TOBIAS: Add camera-specific embeddings to three used feature maps.

        # dimensions are [batchsize, feature channels, h, w]
        # num channels [256, 512, 1024, 2048]

        # only last three dims are used.
        for i, feat in enumerate(features[-3:]):
            # Expand cam_embedder to match last two dimensions of feat, too.
            expanded_embedding = self.cam_embedder[i].expand(
                N_CAMS, feat.tensors.size(1),
                feat.tensors.size(2),
                feat.tensors.size(3)
            ).to(device=features[i + 1].tensors.device)
            
            # Add the positional encoding (first features element unused)
            features[i + 1].tensors = features[i + 1].tensors + expanded_embedding
            
        #-----------------------------------------------------------------------
        # TOBIAS: move subsequent cam tokens from batch slot to width slot

        for feat_map_idx in range(0, 4):
            ts = features[feat_map_idx].tensors.shape
            features[feat_map_idx].tensors = features[feat_map_idx].tensors.reshape(1, ts[1], ts[2], ts[3] * ts[0])
            ms = features[feat_map_idx].mask.shape
            features[feat_map_idx].mask = features[feat_map_idx].mask.reshape(1, ms[1], ms[2] * ms[0])
            ps = pos[feat_map_idx].shape
            pos[feat_map_idx] = pos[feat_map_idx].reshape(1, ps[1], ps[2], ps[3], ps[4] * ps[0])
        #-----------------------------------------------------------------------

        features_all = features
        # pos_all = pos
        # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        features = features[-3:]
        # pos = pos[-3:]

        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-3:]

        # srcs = []
        # masks = []
        src_list = []
        mask_list = []
        pos_list = []
        # for l, (feat, prev_feat) in enumerate(zip(features, prev_features)):

        frame_features = [prev_features, features]
        if not self.multi_frame_attention:
            frame_features = [features]

        for frame, frame_feat in enumerate(frame_features):
            if self.multi_frame_attention and self.multi_frame_encoding:
                pos_list.extend([p[:, frame] for p in pos[-3:]])
            else:
                pos_list.extend(pos[-3:])

            # src, mask = feat.decompose()

            # prev_src, _ = prev_feat.decompose()

            for l, feat in enumerate(frame_feat):
                src, mask = feat.decompose()

                if self.merge_frame_features:
                    prev_src, _ = prev_features[l].decompose()
                    src_list.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
                else:
                    src_list.append(self.input_proj[l](src))

                mask_list.append(mask)

            # if hasattr(self, 'merge_features'):
            #     srcs.append(self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1)))
            # else:
            #     srcs.append(self.input_proj[l](src))

            # masks.append(mask)
                assert mask is not None

            if self.num_feature_levels > len(frame_feat):
                _len_srcs = len(frame_feat)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        # src = self.input_proj[l](frame_feat[-1].tensors)
                        # if hasattr(self, 'merge_features'):
                        #     src = self.merge_features[l](torch.cat([self.input_proj[l](features[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        # else:
                        #     src = self.input_proj[l](features[-1].tensors)

                        if self.merge_frame_features:
                            src = self.merge_features[l](torch.cat([self.input_proj[l](frame_feat[-1].tensors), self.input_proj[l](prev_features[-1].tensors)], dim=1))
                        else:
                            src = self.input_proj[l](frame_feat[-1].tensors)
                    else:
                        src = self.input_proj[l](src_list[-1])
                        # src = self.input_proj[l](srcs[-1])
                    # m = samples.mask
                    _, m = frame_feat[0].decompose()
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    src_list.append(src)
                    mask_list.append(mask)
                    if self.multi_frame_attention and self.multi_frame_encoding:
                        pos_list.append(pos_l[:, frame])
                    else:
                        pos_list.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(src_list, mask_list, pos_list, query_embeds, targets)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in src_list:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(
                batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        memory = memory_slices
        # memory = memory_slices[-1]
        # features = [NestedTensor(memory_slide) for memory_slide in memory_slices]

        return out, targets, features_all, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class DeformablePostProcess(PostProcess):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, view, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        #-----------------------------------------------------------------------
        # TOBIAS: I train on xywh (not cxcy) but need xyxy for eval 
        boxes = out_bbox
        #boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        #boxes = bbox_xywh_to_xyxy(out_bbox)

        boxes = inverse_min_max_scaling(boxes)
        rvec, tvec = load_spec_extrinsics(view)
        camera_matrix, _ = load_spec_intrinsics(view)

        # ignore batch dimension
        boxes_reshaped = torch.squeeze(boxes, dim=0)
        boxes_reshaped = transform_3D_cylinder_to_2D_COCO_bbox_params(
            cylinder=boxes_reshaped,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            device=boxes.device
        )
        boxes = boxes_reshaped.unsqueeze(dim=0)

        boxes = bbox_xywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        #img_h, img_w = target_sizes.unbind(1)
        #scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        #boxes = boxes * scale_fct[:, None, :]
        #-----------------------------------------------------------------------

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
    
    def process_cylinders(self, boxes, view):
        """Project cylinders to bbox in XYXY format."""

        boxes = inverse_min_max_scaling(boxes)
        rvec, tvec = load_spec_extrinsics(view)
        camera_matrix, _ = load_spec_intrinsics(view)
        boxes = transform_3D_cylinder_to_2D_COCO_bbox_params(
            cylinder=boxes,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            device=boxes.device
        )
        boxes = bbox_xywh_to_xyxy(boxes)

        return boxes
