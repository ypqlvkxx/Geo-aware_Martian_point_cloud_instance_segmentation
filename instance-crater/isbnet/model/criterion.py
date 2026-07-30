from sunau import Au_read
import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.model.matcher import HungarianMatcher
from .model_utils import batch_giou_corres, giou_aabb
import numpy as np

@torch.no_grad()
def get_iou(inputs, targets, thresh=0.5):
    inputs_bool = inputs.detach().sigmoid()
    inputs_bool = inputs_bool >= thresh

    intersection = (inputs_bool * targets).sum(-1)
    union = inputs_bool.sum(-1) + targets.sum(-1) - intersection

    iou = intersection / (union + 1e-6)

    return iou


def compute_dice_loss(inputs, targets, num_boxes, weight, mask=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()

    if mask is not None:
        inputs = inputs * mask
        targets = targets * mask

    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss * weight
    return loss.sum() / (num_boxes + 1e-6)


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mask=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    if mask is not None:
        ce_loss = ce_loss * mask

    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


class Criterion(nn.Module):
    def __init__(
        self,
        semantic_classes=20,
        instance_classes=18,
        semantic_weight=None,
        ignore_label=-100,
        eos_coef=0.1,
        semantic_only=True,
        total_epoch=40,
        trainall=False,
        voxel_scale=50,
    ):
        super(Criterion, self).__init__()

        self.matcher = HungarianMatcher()
        self.semantic_only = semantic_only

        self.ignore_label = ignore_label

        self.label_shift = semantic_classes - instance_classes
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.eos_coef = eos_coef
        self.voxel_scale = voxel_scale

        self.total_epoch = total_epoch

        self.trainall = trainall

        empty_weight = torch.ones(self.instance_classes + 1)
        empty_weight[-1] = self.eos_coef
        if self.semantic_weight:
            for i in range(0, self.instance_classes + self.label_shift):
                empty_weight[i] = self.semantic_weight[i]

        self.register_buffer("empty_weight", empty_weight)


        self.loss_weight = {
            "dice_loss": 1,
            "bce_loss": 1,
            "cls_loss": 0.0,
            "entropy_loss": 0.5,
            "iou_loss": 0.5,
            "box_loss": 0.5,
            "center_loss": 0.5,
            "area_loss": 0.0,
            "giou_loss": 0.5,
            "var_loss": 1.0, 
            "dist_loss": 0.05,
            "KL_loss": 0.0,
        }

    def area_consistency_loss(self, areas, semantic_labels, instance_labels, batch_offsets):
        ins_num = 0
        consistency_loss = 0.0
        for b in range(len(batch_offsets) - 1):
            b_s, b_e = batch_offsets[b], batch_offsets[b + 1]
            ins_lb = torch.unique(instance_labels[b_s:b_e])
            unique_sem_label = torch.unique(semantic_labels[b_s:b_e])
            
            for sem_label in unique_sem_label:
                sem_mask = (semantic_labels == sem_label)
                sem_mask[0:b_s] = False; sem_mask[b_e:] = False
                for ins in ins_lb.cpu().numpy():
                    mask = (instance_labels == ins)
                    mask[0:b_s] = False; mask[b_e:] = False
                    
                    final_mask = mask & sem_mask
                    if final_mask.sum() > 0:
  
                        ins_area = areas[final_mask]
                        if ins_area.shape[0] > 1:  # 至少两个点才计算方差
                            variance = torch.var(ins_area, unbiased=False)
                            ins_num += 1
                            consistency_loss += variance
        if ins_num > 0:
            consistency_loss = consistency_loss / ins_num
        return consistency_loss
                        
                                
    def cal_point_wise_loss(
        self,
        semantic_scores,
        centroid_offset,
        corners_offset,
        box_conf,
        areas,
        semantic_labels,
        instance_labels,
        centroid_offset_labels,
        corners_offset_labels,
        coords_float,
        dc_mask_features,
        dc_ins_label,
        dc_semantic_labels,
        pw_area_labels,
        batch_offsets,
        weights,
        dc_batch_offsets
    ):
        losses = {}

        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device=semantic_labels.device)
        else:
            weight = None

        logits = semantic_scores.transpose(1, 2).reshape(-1, 2)#19)#这里的19原来是dataset.numclass
        
        ignored_bool = (semantic_labels == -1)

        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = semantic_labels[valid_idx]
        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, 2).long().to('cuda')
        inserted_value = torch.zeros((1,)).long().to('cuda')
        
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        
        import numpy as np
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
        semantic_loss = self.criterion(valid_logits, valid_labels).mean()
    
        losses["pw_sem_loss"] = semantic_loss

         
        if not self.semantic_only:
            self.ignore_label = -100
            pos_inds = instance_labels != self.ignore_label
            #pos_inds = semantic_labels != self.ignore_label
            total_pos_inds = pos_inds.sum()
            if total_pos_inds == 0:
                offset_loss = 0 * centroid_offset.sum()
                offset_x_loss = 0 * centroid_offset.sum()
                offset_y_loss = 0 * centroid_offset.sum()
                offset_z_loss = 0 * centroid_offset.sum()
                offset_vertices_loss = 0 * corners_offset.sum()
                conf_loss = 0 * box_conf.sum()
                giou_loss = 0 * box_conf.sum()
            else:
                
                offset_loss = F.l1_loss(centroid_offset[pos_inds], centroid_offset_labels[pos_inds],reduction="none")*weights[pos_inds]
                offset_loss = torch.mean(torch.mean(offset_loss,dim=0))
                #box的偏移预测
                offset_vertices_loss = F.l1_loss(corners_offset[pos_inds], corners_offset_labels[pos_inds], reduction="none")*weights[pos_inds]
                offset_vertices_loss = torch.mean(torch.mean(offset_vertices_loss,dim=0))

                iou_gt, giou = batch_giou_corres(
                    corners_offset[pos_inds] + coords_float[pos_inds].repeat(1, 2),
                    corners_offset_labels[pos_inds] + coords_float[pos_inds].repeat(1, 2),
                )
                giou_loss = torch.mean((1 - giou)*weights[pos_inds][:,0])

                iou_gt = iou_gt.detach()
                conf_loss = F.mse_loss(box_conf[pos_inds], iou_gt, reduction="none")*weights[pos_inds][:,0]
                conf_loss = torch.mean(conf_loss)

                area_loss = (F.mse_loss(areas[pos_inds],pw_area_labels[pos_inds], reduction="none")/((pw_area_labels[pos_inds]*pw_area_labels[pos_inds])+1e-6))*weights[pos_inds]
                area_loss = torch.tensor(0.0, requires_grad=True, device=dc_mask_features.device, dtype=torch.float)#torch.mean(area_loss)

                area_con_loss = torch.tensor(0.0, requires_grad=True, device=dc_mask_features.device, dtype=torch.float)#self.area_consistency_loss(areas, semantic_labels, instance_labels, batch_offsets)

        if not self.semantic_only:
            losses["pw_center_loss"] = offset_loss * self.voxel_scale / 50.0
            losses["pw_corners_loss"] = offset_vertices_loss * self.voxel_scale / 50.0
            losses["pw_giou_loss"] = giou_loss
            losses["pw_conf_loss"] = conf_loss * 100
            losses["pw_area_loss"] = area_loss  / 100.0
            losses["pw_area_con_loss"] = area_con_loss * 100

            #拉力loss
            dc_mask_features = dc_mask_features.squeeze(2)
            dc_ins_label = dc_ins_label
            dc_semantic_labels = dc_semantic_labels
            dc_batch_offsets = dc_batch_offsets     
                    
            losses["pw_var_loss"] = torch.tensor(0.0, requires_grad=True, device=dc_mask_features.device, dtype=torch.float)
            losses["pw_dist_loss"] = torch.tensor(0.0, requires_grad=True, device=dc_mask_features.device, dtype=torch.float)
            losses["pw_reg_loss"] = torch.tensor(0.0, requires_grad=True, device=dc_mask_features.device, dtype=torch.float)
            ins_num = 0
            
            for b in range(len(dc_batch_offsets) - 1):
                b_s, b_e = dc_batch_offsets[b], dc_batch_offsets[b + 1]
                ins_lb = torch.unique(dc_ins_label[b_s:b_e])
                unique_sem_label = torch.unique(dc_semantic_labels[b_s:b_e])
                center_mask_features_list = []
                var_weights_list = []
                
                all_ins_pt_num = (dc_ins_label[b_s:b_e] != -100).sum()

                for sem_label in unique_sem_label:
                    if sem_label != 0:
                        sem_mask = (dc_semantic_labels == sem_label)
                        sem_mask[0:b_s] = False; sem_mask[b_e:] = False
                        for ins in ins_lb.cpu().numpy():
                            if ins != -100:
                                mask = (dc_ins_label == ins)
                                mask[0:b_s] = False; mask[b_e:] = False
                                
                                final_mask = mask & sem_mask
                                if final_mask.sum() > 0:
                                    ins_pt_num = final_mask.sum()
                                    var_weights = torch.log(all_ins_pt_num / ins_pt_num)
                                    ins_num += 1
                                    ins_mask_features = dc_mask_features[final_mask]
                                    mean_ = torch.mean(ins_mask_features,dim = 0)
                        
                                    center_mask_features_list.append(mean_)  
                                    var_weights_list.append(var_weights)  

                                    losses["pw_var_loss"] = losses["pw_var_loss"] + var_weights * torch.mean(torch.norm((ins_mask_features - mean_), dim=1),dim =0)

                N_ins = len(center_mask_features_list)
                distance_matrix = torch.zeros((N_ins, N_ins))
                losses["pw_reg_loss"] = losses["pw_reg_loss"] + torch.mean(torch.norm(torch.stack(center_mask_features_list),dim = 1),dim = 0)
                for i in range(N_ins):
                    for j in range(i + 1, N_ins):
                        distance_matrix[i, j] = (1/var_weights_list[i]) * torch.norm(center_mask_features_list[i] - center_mask_features_list[j], dim=0, p=2)
                        distance_matrix[j, i] = distance_matrix[i, j]  # 因为距离是对称的
                if N_ins == 1:
                    losses["pw_dist_loss"] = losses["pw_dist_loss"] + distance_matrix[~torch.eye(N_ins, dtype=torch.bool)].sum()
                else:
                    losses["pw_dist_loss"] = losses["pw_dist_loss"] + distance_matrix[~torch.eye(N_ins, dtype=torch.bool)].sum() / (N_ins * (N_ins - 1) // 2)

                
            losses["pw_var_loss"] = 1 * losses["pw_var_loss"] / ins_num
            losses["pw_dist_loss"] = losses["pw_dist_loss"] / (len(dc_batch_offsets) - 1)
            losses["pw_dist_loss"] = 0.1 / (losses["pw_dist_loss"] + 1e-6)
            

        return losses

    def single_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        conf_logits,
        box_preds,
        center_preds,
        area_preds,
        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        center_labels,
        area_labels,
        query_feats_1,
        query_inds1,
        query_inds2,
        voxel_instance_labels,
        sem_logits,
        ins_weights,
        batch_size,
    ):
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)

        num_gt = 0
        ins_num = 0
        for b in range(batch_size):
            mask_logit_b = mask_logits_list[b]
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]
            center_preds_b = center_preds[b]
            area_preds_b = area_preds[b]

            sem_logit_b = sem_logits[b]

            pred_inds, cls_label, inst_label, box_label, center_label, area_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b], center_labels[b], area_labels[b]
            weight = ins_weights[b].to(cls_logits.device)

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue

            mask_logit_pred = mask_logit_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]
            center_pred = center_preds_b[pred_inds]
            area_pred = area_preds_b[pred_inds]
            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch

            loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                mask_logit_pred, inst_label, num_gt_batch, weight=weight
            )
            

            bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
            bce_loss = bce_loss * weight.unsqueeze(1)
            bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
            
            loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss

            gt_iou = get_iou(mask_logit_pred, inst_label)

            loss_dict["iou_loss"] = (
                loss_dict["iou_loss"] + torch.mean(F.mse_loss(conf_logits_pred, gt_iou, reduction="none")*weight)
            )

            target_classes = (
                torch.ones((n_queries), dtype=torch.int64, device=cls_logits.device) * self.instance_classes
            )

            target_classes[pred_inds] = cls_label
            self.criterion_1 = nn.CrossEntropyLoss(weight=self.empty_weight, reduction='none')
            loss_dict["cls_loss"] = loss_dict["cls_loss"] + self.criterion_1(cls_logit_b, target_classes).mean()

            #
            output_reshaped = cls_logit_b.softmax(-1)
            probs = output_reshaped 
            entropy_re_loss = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)#torch.mean(entropy_loss)  
            loss_dict["entropy_loss"] = loss_dict["entropy_loss"] + entropy_re_loss

            ###KL散度计算
            x_log = F.log_softmax(cls_logit_b,dim=1)
            y = F.softmax(sem_logit_b,dim=1)
            self.klloss = nn.KLDivLoss(reduction='mean')
            loss_dict["KL_loss"] = loss_dict["KL_loss"] + self.klloss(x_log, y)

            loss_dict["box_loss"] = (
                loss_dict["box_loss"]
                + (self.voxel_scale / 50.0) * torch.mean(F.l1_loss(box_pred, box_label, reduction="none")*weight.unsqueeze(1))
            )
            loss_dict["center_loss"] = (
                loss_dict["center_loss"]
                + (self.voxel_scale / 50.0) * torch.mean(F.l1_loss(center_pred, center_label, reduction="none")*weight.unsqueeze(1))
            )
            loss_dict["area_loss"] = (
                loss_dict["area_loss"]
                + (1 / 100000.0) * torch.mean((F.mse_loss(area_pred, area_label, reduction="none")/(area_label*area_label))*weight.unsqueeze(1))
            )

            iou_gt, giou = giou_aabb(box_pred, box_label, coords=None)

            loss_dict["giou_loss"] = loss_dict["giou_loss"] + torch.sum(1 - giou*weight) / num_gt_batch
            
            if not self.semantic_only:
                #拉力和推力loss计算
                query_inds_b = query_inds1[b]
                query_inds_b_2 = query_inds2[b]
                query_feats_b = query_feats_1[b]
                ins_label_b = voxel_instance_labels[query_inds_b.long()]
                ins_label_b = ins_label_b[query_inds_b_2.long()]
                center_mask_features_list = []
                
                ins_lb = torch.unique(ins_label_b)
                for ins in ins_lb.cpu().numpy():
                    if ins != -100:
                        ins_num += 1
                        mask = (ins_label_b == ins)
                        ins_mask_features = query_feats_b[:,mask]
                        mean_ = torch.mean(ins_mask_features,dim = 1)
            
                        center_mask_features_list.append(mean_)    
                        loss_dict["var_loss"] = loss_dict["var_loss"] + torch.mean(torch.norm((ins_mask_features - mean_[:,np.newaxis]), dim=0),dim =0)

                N_ins = len(center_mask_features_list)
                distance_matrix = torch.zeros((N_ins, N_ins))
                
                for i in range(N_ins):
                    for j in range(i + 1, N_ins):
                        distance_matrix[i, j] = torch.norm(center_mask_features_list[i] - center_mask_features_list[j], dim=0, p=2)
                        distance_matrix[j, i] = distance_matrix[i, j]  # 因为距离是对称的
                if N_ins == 1:
                    loss_dict["dist_loss"] = loss_dict["dist_loss"] + distance_matrix[~torch.eye(N_ins, dtype=torch.bool)].sum()
                else:
                    loss_dict["dist_loss"] = loss_dict["dist_loss"] + distance_matrix[~torch.eye(N_ins, dtype=torch.bool)].sum() / (N_ins * (N_ins - 1) // 2)
        
        if not self.semantic_only:
            loss_dict["var_loss"] = 1 * loss_dict["var_loss"] / ins_num #
            loss_dict["dist_loss"] = loss_dict["dist_loss"] / batch_size # 
            loss_dict["dist_loss"]= 0.01 / loss_dict["dist_loss"] #

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / batch_size

        return loss_dict

    def single_aux_layer_loss(
        self,
        cls_logits,
        mask_logits_list,
        conf_logits,
        box_preds,
        row_indices,
        cls_labels,
        inst_labels,
        box_labels,
        ins_weights,
        batch_size,
    ):
        loss_dict = {}

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=cls_logits.device, dtype=torch.float)

        num_gt = 0
        for b in range(batch_size):
            mask_logit_b = mask_logits_list[b]
            cls_logit_b = cls_logits[b]  # n_queries x n_classes
            conf_logits_b = conf_logits[b]  # n_queries
            box_preds_b = box_preds[b]

            pred_inds, cls_label, inst_label, box_label = row_indices[b], cls_labels[b], inst_labels[b], box_labels[b]
            weight = ins_weights[b].to(cls_logits.device)

            n_queries = cls_logit_b.shape[0]

            if mask_logit_b is None:
                continue

            if pred_inds is None:
                continue

            mask_logit_pred = mask_logit_b[pred_inds]
            conf_logits_pred = conf_logits_b[pred_inds]
            box_pred = box_preds_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch

            loss_dict["dice_loss"] = loss_dict["dice_loss"] + compute_dice_loss(
                mask_logit_pred, inst_label, num_gt_batch, weight=weight
            )
            

            bce_loss = F.binary_cross_entropy_with_logits(mask_logit_pred, inst_label, reduction="none")
            bce_loss = bce_loss * weight.unsqueeze(1)
            bce_loss = bce_loss.mean(1).sum() / (num_gt_batch + 1e-6)
            
            loss_dict["bce_loss"] = loss_dict["bce_loss"] + bce_loss

            gt_iou = get_iou(mask_logit_pred, inst_label)

            loss_dict["iou_loss"] = (
                loss_dict["iou_loss"] + torch.mean(F.mse_loss(conf_logits_pred, gt_iou, reduction="none")*weight)
            )

            target_classes = (
                torch.ones((n_queries), dtype=torch.int64, device=cls_logits.device) * self.instance_classes
            )

            target_classes[pred_inds] = cls_label

            self.criterion_1 = nn.CrossEntropyLoss(weight=self.empty_weight, reduction='none')
            loss_dict["cls_loss"] = loss_dict["cls_loss"] + self.criterion_1(cls_logit_b, target_classes).mean()

            loss_dict["box_loss"] = (
                loss_dict["box_loss"]
                + (self.voxel_scale / 50.0) * torch.mean(F.l1_loss(box_pred, box_label, reduction="none")*weight.unsqueeze(1))
            )

            iou_gt, giou = giou_aabb(box_pred, box_label, coords=None)

            loss_dict["giou_loss"] = loss_dict["giou_loss"] + torch.sum(1 - giou*weight) / num_gt_batch

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] / batch_size

        return loss_dict

    def forward(self, batch_inputs, model_outputs):
        loss_dict = {}

        semantic_labels = batch_inputs["semantic_labels"]
        instance_labels = batch_inputs["instance_labels"]

        if model_outputs is None:
            loss_dict["Placeholder"] = torch.tensor(
                0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float
            )

            return loss_dict

        if self.semantic_only or self.trainall:
            # '''semantic loss'''
            #semantic_scores = model_outputs["semantic_scores"]
            semantic_scores = model_outputs["logits"]
            centroid_offset = model_outputs["centroid_offset"]
            corners_offset = model_outputs["corners_offset"]
            box_conf = model_outputs["box_conf"]
            areas = model_outputs["areas"]
            

            coords_float = batch_inputs["coords_float"]
            centroid_offset_labels = batch_inputs["centroid_offset_labels"]
            corners_offset_labels = batch_inputs["corners_offset_labels"]


            if not self.semantic_only:
                batch_offsets = model_outputs["batch_offsets"]
                dc_mask_features = model_outputs["dc_mask_features"]
                dc_ins_label = model_outputs["dc_ins_label"]
                dc_semantic_labels = model_outputs["dc_semantic_labels"]
                dc_batch_offsets = model_outputs["dc_batch_offsets"]
                pw_area_labels = model_outputs["pw_area_labels"]
                weights = model_outputs["weights"]
            else:
                batch_offsets = torch.tensor(0.0, requires_grad=False)
                dc_mask_features = torch.tensor(0.0, requires_grad=False)
                dc_ins_label = torch.tensor(0.0, requires_grad=False)
                dc_semantic_labels = torch.tensor(0.0, requires_grad=False)
                dc_batch_offsets = torch.tensor(0.0, requires_grad=False)
                pw_area_labels = torch.tensor(0.0, requires_grad=False)
                weights = torch.tensor(0.0, requires_grad=False)

            #cross-entropy loss, L1 loss and gIoU loss
            point_wise_loss = self.cal_point_wise_loss(
                semantic_scores,
                centroid_offset,
                corners_offset,
                box_conf,
                areas, 
                semantic_labels,
                instance_labels,
                centroid_offset_labels,
                corners_offset_labels,
                coords_float,
                dc_mask_features,
                dc_ins_label,
                dc_semantic_labels,
                pw_area_labels,
                batch_offsets,
                weights,
                dc_batch_offsets
            )

            loss_dict.update(point_wise_loss)

            if self.semantic_only:
                return loss_dict

        for k in loss_dict.keys():
            if "pw" in k:
                loss_dict[k] = loss_dict[k] * 0.25

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float)
            loss_dict["aux_" + k] = torch.tensor(
                0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float
            )

        """ Main loss """
        cls_logits = model_outputs["cls_logits"]
        mask_logits = model_outputs["mask_logits"]
        conf_logits = model_outputs["conf_logits"]
        box_preds = model_outputs["box_preds"]
        center_preds = model_outputs["center_preds"]
        area_preds = model_outputs["area_preds"]

        dc_inst_mask_arr = model_outputs["dc_inst_mask_arr"]

        batch_size, n_queries = cls_logits.shape[:2]

        gt_dict, aux_gt_dict = self.matcher.forward_dup(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            dc_inst_mask_arr,
            dup_gt=4,
        )

        # NOTE main loss

        row_indices = gt_dict["row_indices"]
        inst_labels = gt_dict["inst_labels"]
        cls_labels = gt_dict["cls_labels"]
        box_labels = gt_dict["box_labels"]
        center_labels = gt_dict["center_labels"]
        area_labels = gt_dict["area_labels"]
        ins_weights = gt_dict["weights"]

        if not self.semantic_only:
            query_feats_1 = model_outputs["query_feats_1"]
            query_inds1 = model_outputs["query_inds1"]
            query_inds2 = model_outputs["query_inds2"]
            voxel_instance_labels = model_outputs["voxel_instance_labels"]
            sem_logits = model_outputs["sem_logits"]
        else:
            query_feats_1 = torch.tensor(0.0, requires_grad=False)
            query_inds1 = torch.tensor(0.0, requires_grad=False)
            query_inds2 = torch.tensor(0.0, requires_grad=False)
            voxel_instance_labels = torch.tensor(0.0, requires_grad=False)
            sem_logits = torch.tensor(0.0, requires_grad=False)

        #voxel_coords_float = model_outputs["voxel_coords_float"]
            
        main_loss_dict = self.single_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            center_preds,
            area_preds,
            row_indices,
            cls_labels,
            inst_labels,
            box_labels,
            center_labels,
            area_labels,
            query_feats_1,
            query_inds1,
            query_inds2,
            voxel_instance_labels,
            sem_logits,
            ins_weights,
            batch_size,
        )

        for k, v in self.loss_weight.items():
            loss_dict[k] = loss_dict[k] + main_loss_dict[k] * v

        # NOTE aux loss

        aux_row_indices = aux_gt_dict["row_indices"]
        aux_inst_labels = aux_gt_dict["inst_labels"]
        aux_cls_labels = aux_gt_dict["cls_labels"]
        aux_box_labels = aux_gt_dict["box_labels"]
        aux_ins_weights = aux_gt_dict["weights"]

        aux_main_loss_dict = self.single_aux_layer_loss(
            cls_logits,
            mask_logits,
            conf_logits,
            box_preds,
            aux_row_indices,
            aux_cls_labels,
            aux_inst_labels,
            aux_box_labels,
            aux_ins_weights,
            batch_size,
        )

        coef_aux = 2.0
        for k, v in self.loss_weight.items():
            loss_dict["aux_" + k] = loss_dict["aux_" + k] + aux_main_loss_dict[k] * v * coef_aux

        return loss_dict
