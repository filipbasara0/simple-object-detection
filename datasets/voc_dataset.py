import os

import pandas as pd
import numpy as np
from datasets import data_aug as T


class PascalVOCDataset:

    def __init__(self):
        root_dir = "./data/"
        self.img_dir = os.path.join(root_dir, "VOC2012", "JPEGImages")

    def load_data(self, dataset_path, labels):
        instances = []
        data_df = pd.read_pickle(dataset_path)
        for _, row in data_df.iterrows():
            img_path = row["filename"]
            labels_ = row["labels"]

            image_path = f"{self.img_dir}/{img_path}"

            labels_ = [[labels.index(l)] for l in labels_]

            targets_ = np.concatenate([row["bboxes"], labels_],
                                      axis=-1).tolist()

            instances.append({"image_path": image_path, "target": targets_})
        return instances

    def get_transform(self):
        return T.Sequence([
            T.RandomHSV(80, 80, 60),
            T.HorizontalFlip(),
            T.RandomScale(0.28),
            T.RandomTranslate(0.28, diff=True),
            T.RandomRotate(20),
            T.RandomShear(0.25)
        ], [0.5, 0.5, 0.5, 0.5, 0.45, 0.45])
