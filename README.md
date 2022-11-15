# Simple Object Detection

![45_valid](https://user-images.githubusercontent.com/29043871/201867189-99262208-45e8-492a-b77e-306c5b03b12e.jpg)
![70_valid](https://user-images.githubusercontent.com/29043871/201868422-a7137139-41cf-444d-aa5a-98364f0e14c1.jpg)

A simple yet effective object detection repository.

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
from inference import load_model, load_image
from datasets import reverse_transform_classes
from utils import draw_bboxes

image = load_image("path/to/img.jpg", image_size=480)
preds = predictor(image)
bboxes = preds["predicted_boxes"][0]
scores = preds["scores"][0]
classes = reverse_transform_classes(preds["pred_classes"], "pascal_voc_2012")[0]

image = image[0].permute(1, 2, 0).detach().cpu().numpy()
draw_bboxes(f"./path/to/pred.jpg", image, bboxes, scores, classes)
```

### Create your own Dataset

To add a new dataset, create a new file `datasets/my_dataset.py`. In `datasets/my_dataset.py`, you should specify training transforms for augmentation (can be `None` if you don't need them) and a function that loads the data in the following format:

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
- `data_fn` - function that loads and converts the data to the above format
- `transform_fn` - returns transforms that are used for augmentation during training; set to `None` if you don't want augmentation
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

![40_valid](https://user-images.githubusercontent.com/29043871/201866833-3210adbf-8c4d-4801-9d74-e3b9399cdf07.jpg)
![69_valid](https://user-images.githubusercontent.com/29043871/201866945-181e1611-9e89-4f38-af3a-961490c3cdd7.jpg)
![77_valid](https://user-images.githubusercontent.com/29043871/201868890-2183e6a4-7d68-4f5c-a7c4-996e3c4b6301.jpg)
![11_valid](https://user-images.githubusercontent.com/29043871/201869177-37929c3d-a4b8-414e-9f64-51a6f53569dc.jpg)

### Carla Traffic Lights

Model with the same specification as above was trained for 50 epochs and obtained a mAP of 60 on a small eval dataset.
Training took 3 hours on a GTX1070Ti.

Dataset collected by myself in the CARLA simulator can be found [here](https://drive.google.com/drive/folders/1TXkPLWlNgauPhQnKEoPDZsx7Px1MD9n_?usp=sharing), annotations can be found [here](https://github.com/affinis-lab/traffic-light-detection-module/blob/master/dataset/carla_all.csv).

Pretrained model can be found [here](https://drive.google.com/file/d/17mcQ-Ct6bUTS8BEpeDjaZMIFmHS2gptl/view?usp=share_link)

![30_valid](https://user-images.githubusercontent.com/29043871/201873987-5f599152-e55f-4b61-afa3-0a9954813f6a.jpg)
![pred (1)](https://user-images.githubusercontent.com/29043871/201874722-34fe2f55-80d7-43ee-8249-53ac5b891645.jpg)
![1_valid](https://user-images.githubusercontent.com/29043871/201874420-cf1bf086-8cf4-4d0e-9bca-17fc68051ee2.jpg)
![pred (2)](https://user-images.githubusercontent.com/29043871/201875505-7498bb33-e77f-410a-97ef-00d5a2cde769.jpg)
![loooong (1)](https://user-images.githubusercontent.com/29043871/201876451-c862f5bf-6302-4937-8ca0-08b83eea0c84.jpg)
![occlusion](https://user-images.githubusercontent.com/29043871/201880563-654409d5-4d5c-4c32-a7ba-ea4f62c4eef9.jpg)

Amazingly, the model can even detect IRL traffic lights (although with a lower confidence):

![carla-irl-1 (1)](https://user-images.githubusercontent.com/29043871/201879893-18291a92-fcfe-418e-a3c1-0cfd54960e7e.jpg)
![carla-irl-5 (1)](https://user-images.githubusercontent.com/29043871/201879903-f32ee233-d4e2-40b7-a0b4-52492e617a82.jpg)
![carla-irl-3 (1)](https://user-images.githubusercontent.com/29043871/201879898-8eafd86d-e2bc-4ac2-8aeb-5a3242a8bd3e.jpg)
![carla-irl-4 (1)](https://user-images.githubusercontent.com/29043871/201879902-3dc9284f-0f4e-4757-a3bf-f5b4cb6f999b.jpg)

## To Do

- Add support for segmentation
- Add DETR
- Train on COCO (once i manage to get some better hardware)
