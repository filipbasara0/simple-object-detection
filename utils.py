import sys
import os
import logging
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import torch

from datasets import reverse_transform_classes


def draw_bboxes(path, img, bboxes, confidences, classes):
    img = (img * 255).round().astype("uint8").copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    img = np.array(img)
    for (start_x, start_y, end_x,
         end_y), confidence, class_ in zip(bboxes, confidences, classes):
        start_x, start_y, end_x, end_y = int(start_x), int(start_y), int(
            end_x), int(end_y)
        (w, h), baseline = cv2.getTextSize(f"{confidence:.2f} {class_}", font,
                                           font_scale, thickness)
        cv2.rectangle(img, (start_x, start_y - (2 * baseline + 5)),
                      (start_x + w, start_y), (0, 255, 255), -1)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 255),
                      thickness)
        cv2.putText(img, f"{confidence:.2f} {class_}", (start_x, start_y), font,
                    font_scale, (0, 0, 0), thickness)
    cv2.imwrite(path, img[:, :, ::-1])


def update_map(targets, results, metric):
    preds = [
        dict(
            boxes=torch.cat(results["predicted_boxes"], dim=0),
            scores=torch.cat(results["scores"], dim=0),
            labels=torch.cat(results["pred_classes"], dim=0),
        )
    ]
    target = [
        dict(
            boxes=torch.cat(targets, dim=0)[:, :4],
            labels=torch.cat(targets, dim=0)[:, 4],
        )
    ]
    metric.update(preds, target)
    return metric


def log_images(dataset_name, epoch_path, all_images, all_results):
    os.makedirs(epoch_path)
    step = 0
    for images, results in list(zip(all_images, all_results)):
        transformed_classes = reverse_transform_classes(results["pred_classes"],
                                                        dataset_name)
        for image, bboxes, scores, classes in list(
                zip(images, results["predicted_boxes"], results["scores"],
                    transformed_classes)):
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            draw_bboxes(f"{epoch_path}/{step}.jpg", image, bboxes, scores,
                        classes)
            step += 1


class Logger:

    def __init__(self,
                 name,
                 file_path=None,
                 log_size=10 * 1024 * 1024,
                 backup_count=5):
        self.log_size = log_size
        self.backup_count = backup_count
        self._init_logger(name, file_path)

    def log_info(self, message):
        self.logger.info(message)

    def _init_logger(self, name, file_path):
        logging.addLevelName(logging.INFO, "[INF]")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter(
            "%(levelname)s - %(asctime)s - %(message)s")

        if file_path:
            file_handler = RotatingFileHandler(file_path,
                                               maxBytes=self.log_size,
                                               backupCount=self.backup_count)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)
