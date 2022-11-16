import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

from modeling import FCOS, FCOSPredictor
from datasets.data_aug import Resize


def load_model(path, num_classes):
    model_state = torch.load(path)["model_state"]
    model = FCOS(num_classes=num_classes)
    model.load_state_dict(model_state, strict=True)
    return FCOSPredictor(model, num_classes=num_classes)


def load_image(image_path, image_size=480):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image, dtype=np.float64)
    resize = Resize(image_size)
    image, _ = resize(image.copy(), None)
    return F.to_tensor(image.copy()).unsqueeze(0)
