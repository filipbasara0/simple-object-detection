import numpy as np
import torch


def nms(bounding_boxes,
        confidence_scores,
        classes,
        threshold,
        class_agnostic=True):
    device = bounding_boxes.device
    if len(bounding_boxes) == 0:
        return torch.tensor([]).to(device), torch.tensor(
            []).to(device), torch.tensor([]).to(device)

    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    confidence_scores = confidence_scores.detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()

    start_x = bounding_boxes[:, 0]
    start_y = bounding_boxes[:, 1]
    end_x = bounding_boxes[:, 2]
    end_y = bounding_boxes[:, 3]

    picked_boxes = []
    picked_scores = []
    picked_classes = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    order = np.argsort(confidence_scores)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_scores.append(confidence_scores[index])
        picked_classes.append(classes[index])

        order = order[:-1]
        if len(order) == 0:
            break

        x1 = np.maximum(start_x[index], start_x[order])
        x2 = np.minimum(end_x[index], end_x[order])
        y1 = np.maximum(start_y[index], start_y[order])
        y2 = np.minimum(end_y[index], end_y[order])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order] - intersection)

        if not class_agnostic:
            other_classes = classes[order] != classes[index]
            ratio[other_classes] = 0.0

        left = np.where(ratio < threshold)
        order = order[left]

    outputs = [
        torch.tensor(np.array(picked_boxes)).to(device),
        torch.tensor(np.array(picked_scores)).to(device),
        torch.tensor(np.array(picked_classes)).to(device)
    ]

    return outputs
