import torch
from torch import nn

from modeling.losses import sigmoid_focal_loss, IOULoss

INF = 100000000
MAXIMUM_DISTANCES_PER_LEVEL = [-1, 64, 128, 256, 512, INF]


def _match_reg_distances_shape(MAXIMUM_DISTANCES_PER_LEVEL, num_locs_per_level):
    level_reg_distances = []
    for m in range(1, len(MAXIMUM_DISTANCES_PER_LEVEL)):
        level_distances = torch.tensor([
            MAXIMUM_DISTANCES_PER_LEVEL[m - 1], MAXIMUM_DISTANCES_PER_LEVEL[m]
        ],
                                       dtype=torch.float32)
        locs_per_level = num_locs_per_level[m - 1]
        level_distances = level_distances.repeat(locs_per_level).view(
            locs_per_level, 2)
        level_reg_distances.append(level_distances)
    return torch.cat(level_reg_distances, dim=0)


def _calc_bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0] + 1.0) * (bbox[:, 3] - bbox[:, 1] + 1.0)


def _compute_centerness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


def _calculate_reg_targets(xs, ys, bbox_targets):
    l = xs[:, None] - bbox_targets[:, 0][None]
    t = ys[:, None] - bbox_targets[:, 1][None]
    r = bbox_targets[:, 2][None] - xs[:, None]
    b = bbox_targets[:, 3][None] - ys[:, None]
    return torch.stack([l, t, r, b], dim=2)


def _apply_distance_constraints(reg_targets, level_distances):
    max_reg_targets, _ = reg_targets.max(dim=2)
    return torch.logical_and(max_reg_targets >= level_distances[:, None, 0], \
                             max_reg_targets <= level_distances[:, None, 1])


def _prepare_labels(locations, targets_batch):
    device = targets_batch[0].device
    num_locs_per_level = [len(l) for l in locations]
    level_distances = _match_reg_distances_shape(MAXIMUM_DISTANCES_PER_LEVEL,
                                                 num_locs_per_level).to(device)

    all_locations = torch.cat(locations, dim=0).to(device)
    xs, ys = all_locations[:, 0], all_locations[:, 1]

    all_reg_targets = []
    all_cls_targets = []
    for targets in targets_batch:
        bbox_targets = targets[:, :4]
        cls_targets = targets[:, 4]

        reg_targets = _calculate_reg_targets(xs, ys, bbox_targets)

        is_in_boxes = reg_targets.min(dim=2)[0] > 0

        fits_to_feature_level = _apply_distance_constraints(
            reg_targets, level_distances).to(device)

        bbox_areas = _calc_bbox_area(bbox_targets)

        locations_to_gt_area = bbox_areas[None].repeat(len(all_locations), 1)
        locations_to_gt_area[is_in_boxes == 0] = INF
        locations_to_gt_area[fits_to_feature_level == 0] = INF

        loc_min_area, loc_mind_idxs = locations_to_gt_area.min(dim=1)

        reg_targets = reg_targets[range(len(all_locations)), loc_mind_idxs]

        cls_targets = cls_targets[loc_mind_idxs]
        cls_targets[loc_min_area == INF] = 0

        all_cls_targets.append(
            torch.split(cls_targets, num_locs_per_level, dim=0))
        all_reg_targets.append(
            torch.split(reg_targets, num_locs_per_level, dim=0))

    return _match_pred_format(all_cls_targets, all_reg_targets, locations)


def _match_pred_format(cls_targets, reg_targets, locations):
    cls_per_level = []
    reg_per_level = []
    for level in range(len(locations)):
        cls_per_level.append(torch.cat([ct[level] for ct in cls_targets],
                                       dim=0))

        reg_per_level.append(torch.cat([rt[level] for rt in reg_targets],
                                       dim=0))

    return cls_per_level, reg_per_level


def _get_positive_samples(cls_labels, reg_labels, box_cls_preds, box_reg_preds,
                          centerness_preds, num_classes):
    box_cls_flatten = []
    box_regression_flatten = []
    centerness_flatten = []
    labels_flatten = []
    reg_targets_flatten = []
    for l in range(len(cls_labels)):
        box_cls_flatten.append(box_cls_preds[l].permute(0, 2, 3, 1).reshape(
            -1, num_classes))
        box_regression_flatten.append(box_reg_preds[l].permute(0, 2, 3,
                                                               1).reshape(
                                                                   -1, 4))
        labels_flatten.append(cls_labels[l].reshape(-1))
        reg_targets_flatten.append(reg_labels[l].reshape(-1, 4))
        centerness_flatten.append(centerness_preds[l].reshape(-1))

    cls_preds = torch.cat(box_cls_flatten, dim=0)
    cls_targets = torch.cat(labels_flatten, dim=0)
    reg_preds = torch.cat(box_regression_flatten, dim=0)
    reg_targets = torch.cat(reg_targets_flatten, dim=0)
    centerness_preds = torch.cat(centerness_flatten, dim=0)

    pos_inds = torch.nonzero(cls_targets > 0).squeeze(1)

    reg_preds = reg_preds[pos_inds]
    reg_targets = reg_targets[pos_inds]
    centerness_preds = centerness_preds[pos_inds]

    return reg_preds, reg_targets, cls_preds, cls_targets, centerness_preds, pos_inds


class LossEvaluator:

    def __init__(self):
        self.reg_loss_func = IOULoss("giou")
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.cls_loss_func = sigmoid_focal_loss

    def _get_cls_loss(self, cls_preds, cls_targets, total_num_pos):
        cls_loss = self.cls_loss_func(cls_preds, cls_targets.int())
        return cls_loss.sum() / total_num_pos

    def _get_reg_loss(self, reg_preds, reg_targets, centerness_targets):
        sum_centerness_targets = centerness_targets.sum()
        reg_preds = reg_preds.reshape(-1, 4)
        reg_targets = reg_targets.reshape(-1, 4)
        reg_loss = self.reg_loss_func(reg_preds, reg_targets,
                                      centerness_targets)
        return reg_loss / sum_centerness_targets

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             total_num_pos):
        centerness_loss = self.centerness_loss_func(centerness_preds,
                                                    centerness_targets)
        return centerness_loss / total_num_pos

    def _evaluate_losses(self, reg_preds, cls_preds, centerness_preds,
                         reg_targets, cls_targets, centerness_targets,
                         pos_inds):
        total_num_pos = max(pos_inds.new_tensor([pos_inds.numel()]), 1.0)

        cls_loss = self._get_cls_loss(cls_preds, cls_targets, total_num_pos)

        if pos_inds.numel() > 0:
            reg_loss = self._get_reg_loss(reg_preds, reg_targets,
                                          centerness_targets)
            centerness_loss = self._get_centerness_loss(centerness_preds,
                                                        centerness_targets,
                                                        total_num_pos)
        else:
            reg_loss = reg_preds.sum()
            centerness_loss = centerness_preds.sum()

        return reg_loss, cls_loss, centerness_loss

    def __call__(self, locations, preds, targets_batch, num_classes):
        cls_targets, reg_targets = _prepare_labels(locations, targets_batch)

        cls_preds, reg_preds, centerness_preds = preds

        reg_preds, reg_targets, cls_preds, cls_targets, centerness_preds, pos_inds = _get_positive_samples(
            cls_targets, reg_targets, cls_preds, reg_preds, centerness_preds,
            num_classes)

        centerness_targets = _compute_centerness_targets(reg_targets)

        reg_loss, cls_loss, centerness_loss = self._evaluate_losses(
            reg_preds, cls_preds, centerness_preds, reg_targets, cls_targets,
            centerness_targets, pos_inds)

        return cls_loss, reg_loss, centerness_loss
