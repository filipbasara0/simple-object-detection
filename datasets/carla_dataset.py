import pandas as pd
import os

from datasets import data_aug as T

root_dir = "./data"
img_dir = os.path.join(root_dir, "carla_images")


def load_carla_data(path, labels):
    data = pd.read_csv(path, delimiter=",", header=None)

    dataset = {}
    red = 0
    green = 0
    all_image_paths = os.listdir(img_dir)
    for record in data[1:][data.columns[:7]].values:
        tokens = record[5].split(",")

        xmin, ymin, xmax, ymax = float(tokens[1].split(":")[1]), float(tokens[2].split(":")[1]),\
                               float(tokens[3].split(":")[1]), float(tokens[4].split(":")[1].replace("}", ""))
        xmax += xmin
        ymax += ymin

        if "stop" in record[6]:
            obj_class = "stop"
            red += 1
        else:
            obj_class = "go"
            green += 1

        obj_class = labels.index(obj_class)
        obj = (xmin, ymin, xmax, ymax, obj_class)

        image_path = record[0]
        if image_path not in all_image_paths:
            continue

        if image_path in dataset:
            dataset[image_path].append(obj)
        else:
            dataset[image_path] = [obj]

    print("Red light: ", red)
    print("Green light: ", green)

    instances = []
    for key in dataset.keys():
        inst = {}
        inst["image_path"] = f"{img_dir}/{key}"
        inst["target"] = dataset[key]
        instances.append(inst)

    return instances


def get_carla_transform():
    return T.Sequence([
        T.RandomHSV(80, 80, 60),
        T.HorizontalFlip(),
        T.RandomScale(0.2),
        T.RandomTranslate(0.2, diff=True),
        T.RandomRotate(10),
        T.RandomShear(0.2)
    ], [0.5, 0.5, 0.5, 0.5, 0.45, 0.45])
