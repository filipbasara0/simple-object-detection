import torch

from inference.postprocessing import FCOSPostProcessor
from modeling.loss_evaluation import LossEvaluator


def _locations_per_level(h, w, s):
    locs_x = [i for i in range(w)]
    locs_y = [i for i in range(h)]

    locs_x = [s / 2 + x * s for x in locs_x]
    locs_y = [s / 2 + y * s for y in locs_y]
    locs = [(y, x) for x in locs_x for y in locs_y]
    return torch.tensor(locs)


def _compute_locations(features, fpn_strides):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locs = _locations_per_level(h, w, fpn_strides[level]).to(feature.device)
        locations.append(locs)
    return locations


class FCOSPredictor(torch.nn.Module):

    def __init__(self,
                 model,
                 num_classes,
                 fpn_strides=[8, 16, 32, 64, 128],
                 pre_nms_thresh=0.3,
                 pre_nms_top_n=1000,
                 nms_thresh=0.45,
                 fpn_post_nms_top_n=50):
        super(FCOSPredictor, self).__init__()

        self.model = model
        self.le = LossEvaluator()
        self.post_processor = FCOSPostProcessor(
            pre_nms_thresh=pre_nms_thresh,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_thresh,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=num_classes)
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes

    def forward(self, images, targets_batch=None):
        features, box_cls, box_regression, centerness = self.model(images)
        locations = _compute_locations(features, self.fpn_strides)
        outputs = {}
        if targets_batch != None:
            cls_loss, reg_loss, centerness_loss = self.le(
                locations, (box_cls, box_regression, centerness),
                targets_batch,
                num_classes=self.num_classes)
            outputs["cls_loss"] = cls_loss
            outputs["reg_loss"] = reg_loss
            outputs["centerness_loss"] = centerness_loss
            outputs["combined_loss"] = cls_loss + reg_loss + centerness_loss

        image_size = images.shape[-1]
        predicted_boxes, scores, all_classes = self.post_processor(
            locations, box_cls, box_regression, centerness, image_size)

        outputs["predicted_boxes"] = predicted_boxes
        outputs["scores"] = scores
        outputs["pred_classes"] = all_classes
        return outputs
