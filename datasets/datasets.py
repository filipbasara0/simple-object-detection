import torch

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split

from datasets import data_aug as T
from datasets.voc_dataset import get_pascal_transform, load_pascal_data
from datasets.carla_dataset import get_carla_transform, load_carla_data

PASCAL_DATASET_NAME = "pascal_voc_2012"
CARLA_DATASET_NAME = "carla_traffic_lights"

DATASETS = {
    PASCAL_DATASET_NAME: {
        "dataset_path":
            "./data/voc_combined.csv",
        "data_fn":
            load_pascal_data,
        "transform_fn":
            get_pascal_transform,
        "labels": [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'
        ]
    },
    CARLA_DATASET_NAME: {
        "dataset_path": "./data/carla.csv",
        "data_fn": load_carla_data,
        "transform_fn": get_carla_transform,
        "labels": ["__background__", "go", "stop"]
    }
}


def get_dataset_labels(dataset_name):
    return DATASETS[dataset_name]["labels"]


def reverse_transform_classes(pred_classes, dataset_name):
    labels = get_dataset_labels(dataset_name)
    transformed_classes = []
    for classes in pred_classes:
        classes = classes.detach().cpu().tolist()
        classes = [labels[c] for c in classes]
        transformed_classes.append(classes)
    return transformed_classes


def _collate_targets(batch):
    transposed_batch = list(zip(*batch))
    images = transposed_batch[0]
    targets = transposed_batch[1]
    return torch.stack(images), targets


def _make_dataloaders(data,
                      transforms,
                      image_size,
                      train_batch_size=16,
                      eval_batch_size=16,
                      eval_size=0.1):
    train_data, eval_data = train_test_split(data,
                                             test_size=eval_size,
                                             random_state=42)
    train_dataset = ObjDetDataset(train_data, image_size, transforms=transforms)
    eval_dataset = ObjDetDataset(eval_data, image_size, transforms=None)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  collate_fn=_collate_targets)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 collate_fn=_collate_targets)
    return train_dataloader, eval_dataloader


class ObjDetDataset(Dataset):

    def __init__(self, data, image_size, transforms=None):
        image_paths = []
        targets = []
        for instance in data:
            image_paths.append(instance['image_path'])
            targets.append(instance["target"])
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms
        self.image_size = image_size
        self.resize = T.Resize(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        targets = self.targets[idx]

        image = Image.open(image_path).convert("RGB")

        image = np.array(image, dtype=np.float64)
        targets = np.array(targets, dtype=np.float64)

        if self.transforms:
            image, targets = self.transforms(image.copy(), targets.copy())

        image, targets = self.resize(image.copy(), targets.copy())

        image = F.to_tensor(image.copy())

        return image, torch.tensor(targets.tolist())


def dataset_factory(dataset_name,
                    image_size,
                    dataset_path=None,
                    train_batch_size=8,
                    eval_batch_size=8):
    if dataset_name not in [PASCAL_DATASET_NAME, CARLA_DATASET_NAME]:
        raise ValueError(f"Dataset name must be one of {DATASETS.keys()}!")

    dataset = DATASETS[dataset_name]
    dataset_path = dataset[
        "dataset_path"] if dataset_path == None else dataset_path
    data = DATASETS[dataset_name]["data_fn"](dataset_path, dataset["labels"])
    return _make_dataloaders(data,
                             dataset["transform_fn"](),
                             image_size,
                             train_batch_size=train_batch_size,
                             eval_batch_size=eval_batch_size)
