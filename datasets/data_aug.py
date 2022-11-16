import random
import numpy as np
import cv2
from datasets.aug_utils import (rotate_box, get_enclosing_box, letterbox_image,
                                clip_box, rotate_im, get_corners)

import torch


def mixup(images, bboxes, alpha=1.0):
    indices = torch.randperm(len(images))
    shuffled_images = images[indices]
    shuffled_bboxes = [bboxes[i] for i in indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    lam = torch.tensor(lam)

    mixedup_images = lam * images + (1 - lam) * shuffled_images

    mixedup_bboxes = []
    for bbox, s_bbox in zip(bboxes, shuffled_bboxes):
        mixedup_bboxes.append(torch.cat([bbox, s_bbox]))
    return mixedup_images, mixedup_bboxes


class HorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes


class RandomScale(object):

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, bboxes):
        orig_image, orig_bboxes = img.copy(), bboxes.copy()

        img_shape = img.shape

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        bboxes[:, :4] *= [
            resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y
        ]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        y_lim = int(min(resize_scale_y, 1) * img_shape[0])
        x_lim = int(min(resize_scale_x, 1) * img_shape[1])

        canvas[:y_lim, :x_lim, :] = img[:y_lim, :x_lim, :]

        img = canvas
        bboxes = clip_box(bboxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

        if len(bboxes) == 0:
            return orig_image, orig_bboxes

        return img, bboxes


class RandomTranslate(object):

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1

        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bboxes):
        orig_image, orig_bboxes = img.copy(), bboxes.copy()

        #Chose a random digit to scale by
        img_shape = img.shape

        #translate the image

        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape).astype(np.uint8)

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        #change the origin to the top-left corner of the translated box
        orig_box_cords = [
            max(0, corner_y),
            max(corner_x, 0),
            min(img_shape[0], corner_y + img.shape[0]),
            min(img_shape[1], corner_x + img.shape[1])
        ]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y +
                                         img_shape[0]),
                   max(-corner_x, 0):min(img.shape[1], -corner_x +
                                         img_shape[1]), :]
        canvas[orig_box_cords[0]:orig_box_cords[2],
               orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas

        bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

        if len(bboxes) == 0:
            return orig_image, orig_bboxes

        return img, bboxes


class RandomRotate(object):

    def __init__(self, angle=10):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):
        orig_image, orig_bboxes = img.copy(), bboxes.copy()

        angle = random.uniform(*self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        img = rotate_im(img, angle)

        corners = get_corners(bboxes)

        corners = np.hstack((corners, bboxes[:, 4:]))

        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w

        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w, h))

        new_bbox[:, :4] /= [
            scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y
        ]

        bboxes = new_bbox

        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        if len(bboxes) == 0:
            return orig_image, orig_bboxes

        return img, bboxes


class RandomShear(object):

    def __init__(self, shear_factor=0.2):
        self.shear_factor = shear_factor

        if type(self.shear_factor) == tuple:
            assert len(
                self.shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)

        shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):

        shear_factor = random.uniform(*self.shear_factor)

        w, h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = img.shape[1] + abs(shear_factor * img.shape[0])

        bboxes[:,
               [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        img = cv2.resize(img, (w, h))

        scale_factor_x = nW / w

        bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

        return img, bboxes


class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet 
    
    The aspect ratio is maintained. The longer side is resized to the input 
    size of the network, while the remaining space on the shorter side is filled 
    with black color. **This should be the last transform**
    """

    def __init__(self, inp_dim):
        self.inp_dim = inp_dim

    def __call__(self, img, bboxes=None):
        w, h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)

        scale = min(self.inp_dim / h, self.inp_dim / w)
        if bboxes is not None:
            bboxes[:, :4] *= (scale)

        new_w = scale * w
        new_h = scale * h
        inp_dim = self.inp_dim

        del_h = (inp_dim - new_h) / 2
        del_w = (inp_dim - new_w) / 2

        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

        if bboxes is not None:
            bboxes[:, :4] += add_matrix

        img = img.astype(np.uint8)

        return img, bboxes


class RandomHSV(object):

    def __init__(self, hue=None, saturation=None, brightness=None):
        if hue:
            self.hue = hue
        else:
            self.hue = 0

        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0

        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)

        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)

        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)

    def __call__(self, img, bboxes):

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)

        img = img.astype(int)

        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1, 1, 3))

        img = np.clip(img, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)

        return img, bboxes


class Sequence(object):

    def __init__(self, augmentations, probs=1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes
