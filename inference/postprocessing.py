import torch
from inference.nms import nms


def _convert_bbox_xywh(bbox):
    xmin, ymin, xmax, ymax = _split_into_xyxy(bbox)
    bbox = torch.cat((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1), dim=-1)
    return bbox


def _split_into_xyxy(bbox):
    xmin, ymin, w, h = bbox.split(1, dim=-1)
    return (
        xmin,
        ymin,
        xmin + (w - 1).clamp(min=0),
        ymin + (h - 1).clamp(min=0),
    )


def remove_small_boxes(boxlist, min_size):
    xywh_boxes = _convert_bbox_xywh(boxlist)
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


def _clip_to_image(bboxes, image_size):
    h, w = image_size
    bboxes[:, 0].clamp_(min=0, max=h - 1)
    bboxes[:, 1].clamp_(min=0, max=w - 1)
    bboxes[:, 2].clamp_(min=0, max=h - 1)
    bboxes[:, 3].clamp_(min=0, max=w - 1)
    return bboxes


class FCOSPostProcessor(torch.nn.Module):

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes):
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(self, locations, cls_preds, reg_preds,
                                       cness_preds, image_size):
        B, C, _, _ = cls_preds.shape

        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(B, -1, C).sigmoid()
        reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cness_preds = cness_preds.permute(0, 2, 3, 1).reshape(B, -1).sigmoid()

        candidate_inds = cls_preds > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(B, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        cls_preds = cls_preds * cness_preds[:, :, None]

        bboxes = []
        cls_labels = []
        scores = []
        for i in range(B):
            per_cls_preds = cls_preds[i]
            per_candidate_inds = candidate_inds[i]
            per_cls_preds = per_cls_preds[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_reg_preds = reg_preds[i]
            per_reg_preds = per_reg_preds[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_cls_preds, top_k_indices = per_cls_preds.topk(
                    per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_reg_preds = per_reg_preds[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_reg_preds[:, 0],
                per_locations[:, 1] - per_reg_preds[:, 1],
                per_locations[:, 0] + per_reg_preds[:, 2],
                per_locations[:, 1] + per_reg_preds[:, 3],
            ],
                                     dim=1)

            detections = _clip_to_image(detections, (image_size, image_size))
            detections = remove_small_boxes(detections, self.min_size)
            bboxes.append(detections)
            cls_labels.append(per_class)
            scores.append(torch.sqrt(per_cls_preds))

        return bboxes, scores, cls_labels

    def forward(self, locations, cls_preds, reg_preds, cness_preds, image_size):
        sampled_boxes = []
        all_scores = []
        all_classes = []
        for l, o, b, c in list(zip(locations, cls_preds, reg_preds,
                                   cness_preds)):
            boxes, scores, cls_labels = self.forward_for_single_feature_map(
                l, o, b, c, image_size)

            sampled_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(cls_labels)

        all_bboxes = list(zip(*sampled_boxes))
        all_scores = list(zip(*all_scores))
        all_classes = list(zip(*all_classes))

        all_bboxes = [torch.cat(bboxes, dim=0) for bboxes in all_bboxes]
        all_scores = [torch.cat(scores, dim=0) for scores in all_scores]
        all_classes = [torch.cat(classes, dim=0) for classes in all_classes]

        boxes, scores, classes = self.select_over_all_levels(
            all_bboxes, all_scores, all_classes)

        return boxes, scores, classes

    def select_over_all_levels(self, boxlists, scores, classes):
        num_images = len(boxlists)
        all_picked_boxes, all_confidence_scores, all_classes = [], [], []
        for i in range(num_images):
            picked_boxes, confidence_scores, picked_classes = nms(
                boxlists[i], scores[i], classes[i], self.nms_thresh)

            number_of_detections = len(picked_boxes)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                image_thresh, _ = torch.kthvalue(
                    confidence_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = confidence_scores >= image_thresh.item()

                keep = torch.nonzero(keep).squeeze(1)
                picked_boxes, confidence_scores, picked_classes = picked_boxes[
                    keep], confidence_scores[keep], picked_classes[keep]

            keep = confidence_scores >= self.pre_nms_thresh
            picked_boxes, confidence_scores, picked_classes = picked_boxes[
                keep], confidence_scores[keep], picked_classes[keep]

            all_picked_boxes.append(picked_boxes)
            all_confidence_scores.append(confidence_scores)
            all_classes.append(picked_classes)
        return all_picked_boxes, all_confidence_scores, all_classes