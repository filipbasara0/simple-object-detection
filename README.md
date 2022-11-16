# Simple Object Detection

![comb1](https://user-images.githubusercontent.com/29043871/201990619-639dc153-1dff-48c7-bd4b-518ebbc1c51e.png)

A minimal object detection repository.

While reading papers and browsing repos to refresh my computer vision knowledge, i noticed that most object detection repos are complicating and cluttered with code - which makes it difficult to understand how object detection works end to end.

This repo should provide a simple and clear understanding on how to tackle the object detection problem. It's like a minimal template for object detection problems.

The aim was to make it easy to use, understand and customize for your own problems or datasets.

Repo is mostly based on the [FCOS architecture](https://arxiv.org/abs/1904.01355).

**All training was done from scratch, without pretrained models or additional data.**

## Setup

1. `git clone git@github.com:filipbasara0/simple-object-detection.git`
2. create virtual environment: `virtualenv -p python3.8 env`
3. activate virtual environment: `source env/bin/activate`
4. install requirements: `pip install -r requirements.txt`

## Usage

### Training

```
python train.py --resolution=480 --dataset="pascal_voc_2012"   --output_dir="trained_models/model.pth"   --train_batch_size=8 --eval_batch_size=8   --num_epochs=81 --learning_rate=1e-3 --save_model_epochs=1 --num_classes=19 --adam_weight_decay=5e-2
```

### Inference

```python
from inference.load import load_model, load_image
from datasets import reverse_transform_classes
from utils import draw_bboxes

# load a model
predictor = load_model("path/to/model.pth", num_classes=19)

# load an image
image = load_image("path/to/img.jpg", image_size=480)

# obtain results
preds = predictor(image)
bboxes = preds["predicted_boxes"]
scores = preds["scores"]
classes = reverse_transform_classes(preds["pred_classes"], "pascal_voc_2012")

# optional - visualize predictions
image = image[0].permute(1, 2, 0).detach().cpu().numpy()
draw_bboxes(f"./path/to/visualized.jpg", image, bboxes[0], scores[0], classes[0])
```

### Create your own Dataset

To add a new dataset, create a file `datasets/my_dataset.py`. In `datasets/my_dataset.py`, you should create a class that contains two methods - `get_transforms` for training augmentations (can be `None` if you don't need them) and `load_data`:

```python
class MyDataset:

    def load_data(self, dataset_path, labels):
        # load the dataset and return it in the format specified below
        ...

    def get_transforms(self):
        # return transforms (just return None if you don't need any)
        ...
```

`load_data` should return the dataset in the following format:

```python
[
    ...,
    {
        "image_path": "path/to/my/image.jpg",
        "target": [..., [x1,y1,x2,y2,C]]
    }
]
```

x1, y1 and x2,y2 represent top left and bottom right corners of your target bboxes, while C represents a label encoding of your target class `(1,2,...len(C))`. Element 0 is reserved for the `__background__` class, which is used to filter negative samples when preparing the training labels.

Finally, in `datasets/datasets.py` add a new entry to the `DATASETS` dict with thet following fields

- `dataset_path` - path to your dataset metadata (`image_path` and `target`)
- `class_name` - class name for you dataset
- `labels` - list of labels - first element of the list should be the `__background__` class (see Pascal and Carla labels in `datasets/datasets.py`)

## Results

### PascalVOC 2012

Training used extensive data augmentation - random horizontal flipping, scaling, translation, rotation, shearing and HSV. Images were resized to maintain the aspect ratio, using the `letterbox` method.

Additional augmentation such as noise injection, blurring, cropping, (blocks/center) erasing, ... could result in better overall performance.

Backbone architecture is the same as `ConvNext-Tiny`:

- Patch size: `4`
- Layer depths: `[3, 3, 9, 3]`
- Block dims: `[96, 192, 384, 768]`
- Image sizes: `384`, `416` and `480`
- Model resulted in `25M` params

It was trained for 100 epochs and obtained a mAP of 40 on a small eval dataset.
Training took ~30 hours on a GTX1070Ti.

Training bigger models for longer would definitely yield better results.

![comb2](https://user-images.githubusercontent.com/29043871/201991539-072d7c45-faff-4c38-8731-5ce4330c72e1.png)
![comb3](https://user-images.githubusercontent.com/29043871/201994865-4c88a2a7-74eb-4f14-86eb-cd26a951dee4.png)

### Carla Traffic Lights

Model with the same specification as above was trained for 50 epochs and obtained a mAP of 60 on a small eval dataset.
Training took 3 hours on a GTX1070Ti.

Dataset collected by myself in the CARLA simulator can be found [here](https://drive.google.com/drive/folders/1TXkPLWlNgauPhQnKEoPDZsx7Px1MD9n_?usp=sharing), annotations can be found [here](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/dataset/carla_all.csv).

Pretrained model can be found [here](https://drive.google.com/file/d/17mcQ-Ct6bUTS8BEpeDjaZMIFmHS2gptl/view?usp=share_link).

![comb4](https://user-images.githubusercontent.com/29043871/201992324-4323166d-e207-417d-9fe9-8265b885d0fe.png)
![comb5](https://user-images.githubusercontent.com/29043871/201992330-e6929134-b639-4744-9a75-108da64ed033.png)
![comb6](https://user-images.githubusercontent.com/29043871/201992333-f6d32332-b7cd-40c9-a82d-049fe1c567ca.png)

Amazingly, the model can even detect IRL traffic lights (although with a lower confidence):

![comb7](https://user-images.githubusercontent.com/29043871/201992833-011f521c-1acd-44bc-b372-135e44940dbb.png)
![comb8](https://user-images.githubusercontent.com/29043871/201992839-ba3134f2-e86f-49f0-a872-77d4aba980d5.png)

## To Do

- Add support for segmentation
- Add DETR
- Train on COCO (once i manage to get some better hardware)
