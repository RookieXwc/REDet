from __future__ import division

# Standard Library
import copy
import math
from collections.abc import Sequence
import torchvision.transforms as transforms
import numbers
import sys
from PIL import Image

# Import from third library
import cv2
import numpy as np
import random
from pycocotools import mask as maskUtils
from up.data.datasets.transforms import build_transformer
from collections import Counter
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation
)
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY, BATCHING_REGISTRY
from up.utils.general.global_flag import ALIGNED_FLAG
from up.data.datasets.transforms import (
    Augmentation,
    has_gt_bboxes,
    has_gt_ignores,
    has_gt_keyps,
    has_gt_masks,
    has_gt_semantic_seg,
    check_fake_gt
)
from up.data.data_utils import (
    coin_tossing,
    get_image_size,
    is_numpy_image,
    is_pil_image
)

# TODO: Check GPU usage after move this setting down from upper line
cv2.ocl.setUseOpenCL(False)


__all__ = [
    'Flip',
    'KeepAspectRatioResize',
    'KeepAspectRatioResizeMax',
    'BatchPad',
    'ImageExpand',
    'RandomColorJitter',
    'FixOutputResize',
    'ImageCrop',
    'ImageStitchExpand'
]


def tensor2numpy(data):
    if check_fake_gt(data.gt_bboxes) and check_fake_gt(data.gt_ignores):
        return np.zeros((0, 5))
    if data.gt_bboxes is None:
        gts = np.zeros((0, 5))
    else:
        gts = data.gt_bboxes.cpu().numpy()
    if data.gt_ignores is not None:
        igs = data.gt_ignores.cpu().numpy()
    else:
        igs = np.zeros((0, 4))
    if igs.shape[0] != 0:
        ig_bboxes = np.hstack([igs, -np.ones(igs.shape[0])[:, np.newaxis]])
        new_gts = np.concatenate((gts, ig_bboxes))
    else:
        new_gts = gts
    return new_gts


def numpy2tensor(boxes_t):
    ig_bboxes_t = boxes_t[boxes_t[:, 4] == -1][:, :4]
    gt_bboxes_t = boxes_t[boxes_t[:, 4] != -1]
    gt_bboxes_t = torch.as_tensor(gt_bboxes_t, dtype=torch.float32)
    ig_bboxes_t = torch.as_tensor(ig_bboxes_t, dtype=torch.float32)
    if len(ig_bboxes_t) == 0:
        ig_bboxes_t = torch.zeros((1, 4))
    if len(gt_bboxes_t) == 0:
        gt_bboxes_t = torch.zeros((1, 5))
    return gt_bboxes_t, ig_bboxes_t


def np_bbox_iof_overlaps(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:, np.newaxis], 1)


def boxes2polygons(boxes, sample=4):
    num_boxes = len(boxes)
    boxes = np.array(boxes).reshape(num_boxes, -1)
    a = (boxes[:, 2] - boxes[:, 0] + 1) / 2
    b = (boxes[:, 3] - boxes[:, 1] + 1) / 2
    x = (boxes[:, 2] + boxes[:, 0]) / 2
    y = (boxes[:, 3] + boxes[:, 1]) / 2

    angle = 2 * math.pi / sample
    polygons = np.zeros([num_boxes, sample, 2], dtype=np.float32)
    for i in range(sample):
        polygons[:, i, 0] = a * math.cos(i * angle) + x
        polygons[:, i, 1] = b * math.sin(i * angle) + y
    return polygons

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment,
    # wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

@AUGMENTATION_REGISTRY.register('flip')
class Flip(Augmentation):
    """
    """

    def __init__(self, flip_p):
        super(Flip, self).__init__()
        self.flip_p = flip_p

    def augment(self, data):
        if not coin_tossing(self.flip_p):
            return data

        output = copy.copy(data)
        output.image = self.flip_image(data.image)
        height, width = get_image_size(data.image)
        # height, width = data.image.shape[:2]
        if has_gt_bboxes(data):
            output.gt_bboxes = self.flip_boxes(data.gt_bboxes, width)
        if has_gt_ignores(data):
            output.gt_ignores = self.flip_boxes(data.gt_ignores, width)
        if has_gt_keyps(data):
            output.gt_keyps = self.flip_keyps(data.gt_keyps, data.keyp_pairs, width)
        if has_gt_masks(data):
            output.gt_masks = self.flip_masks(data.gt_masks, width)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.flip_semantic_seg(data.gt_semantic_seg)
        output.flipped = True
        return output

    def flip_image(self, img):
        if is_pil_image(img):
            return TF.hflip(img)
        elif is_numpy_image(img):
            return cv2.flip(img, 1)
        else:
            raise TypeError('{} format not supported'.format(type(img)))

    def flip_boxes(self, boxes, width):
        x1 = boxes[:, 0].clone().detach()
        x2 = boxes[:, 2].clone().detach()
        boxes[:, 0] = width - ALIGNED_FLAG.offset - x2
        boxes[:, 2] = width - ALIGNED_FLAG.offset - x1
        return boxes

    def flip_masks(self, polygons, width):
        """Actuall flip polygons"""
        flipped_masks = []
        for polygon in polygons:
            flipped = []
            for poly in polygon:
                p = poly.copy()
                p[0::2] = width - poly[0::2] - ALIGNED_FLAG.offset
                flipped.append(p)
            flipped_masks.append(flipped)
        return flipped_masks

    def flip_keyps(self, keyps, pairs, width):
        if keyps.nelement() == 0:
            return keyps

        N, K = keyps.size()[:2]
        keyps = keyps.view(-1, 3)
        labeled = torch.nonzero(keyps[:, 2] > 0)
        if labeled.size(0) == 0:
            return keyps.view(N, K, 3)

        labeled = labeled.view(1, -1)[0]
        keyps[labeled, 0] = width - ALIGNED_FLAG.offset - keyps[labeled, 0]
        keyps = keyps.view(N, K, 3)
        clone_keyps = keyps.clone().detach()
        for left, right in pairs:
            # keyp_left = keyps[:, left, :].clone()
            # keyps[:, left, :] = keyps[:, right, :]
            # keyps[: right, :] = keyp_left
            keyps[:, left, :] = keyps[:, right, :]
            keyps[:, right, :] = clone_keyps[:, left, :]
        return keyps

    def flip_semantic_seg(self, semantic_seg):
        return self.flip_image(semantic_seg)

@AUGMENTATION_REGISTRY.register('copypaste_flip')
class CopyPasteFlip(Flip):
    """
    """
    def flip_masks(self, masks, width):
        """Actuall flip polygons"""
        masks = masks[0]
        if len(masks) == 0:
            flipped_masks = masks
        else:
            flipped_masks = np.stack([
                cv2.flip(mask, 1)
                for mask in masks
            ])
        return [flipped_masks]


class Resize(Augmentation):
    def __init__(self, scales, max_size=sys.maxsize, separate_wh=False):
        """
        Args:
            scales: evaluable str, e.g. range(500,600,1) or given list
            max_size: int
        """
        super(Augmentation, self).__init__()
        if isinstance(scales, str):
            self.scales = list(eval(scales))
        else:
            assert isinstance(scales, Sequence)
            self.scales = scales
        self.max_size = max_size
        self.separate_wh = separate_wh

    def augment(self, data, scale_factor=None):
        if scale_factor is None:
            # img_h, img_w = data.image.shape[:2]
            img_h, img_w = get_image_size(data.image)
            scale_factor = self.get_scale_factor(img_h, img_w)
        output = copy.copy(data)
        output.image = self.resize_image(data.image, scale_factor)
        if self.separate_wh:
            h, w = get_image_size(data.image)
            o_h, o_w = get_image_size(output.image)
            scale_factor = (o_h / h, o_w / w)
        if has_gt_bboxes(data):
            output.gt_bboxes = self.resize_boxes(data.gt_bboxes, scale_factor)
        if has_gt_ignores(data):
            output.gt_ignores = self.resize_boxes(data.gt_ignores, scale_factor)
        if has_gt_keyps(data):
            output.gt_keyps = self.resize_keyps(data.gt_keyps, scale_factor)
        if has_gt_masks(data):
            output.gt_masks = self.resize_masks(data.gt_masks, scale_factor)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.resize_semantic_seg(data.gt_semantic_seg, scale_factor)
        output.scale_factor = scale_factor
        return output

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        raise NotImplementedError

    def resize_image(self, img, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        if is_pil_image(img):
            origin_w, origin_h = img.size
            target_h, target_w = int(round(ratio_h * origin_h)), int(round(ratio_w * origin_w))
            return TF.resize(img, (target_h, target_w))
        elif is_numpy_image(img):
            return cv2.resize(
                img, None, None,
                fx=ratio_w,
                fy=ratio_h,
                interpolation=cv2.INTER_LINEAR)
        else:
            raise TypeError('{} format not supported'.format(type(img)))

    def resize_boxes(self, boxes, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        if ratio_h == ratio_w:
            boxes[:, :4] *= ratio_h
        else:
            boxes[:, 0] *= ratio_w
            boxes[:, 1] *= ratio_h
            boxes[:, 2] *= ratio_w
            boxes[:, 3] *= ratio_h
        return boxes

    def resize_masks(self, masks, scale_factor):
        """
        Actually resize polygons

        Note:
            Since resizing polygons will cause a shift for masks generation
            here add an offset to eliminate the difference between resizing polygons and resizing masks
        """
        ratio_h, ratio_w = _pair(scale_factor)
        for polys in masks:
            for poly in polys:
                poly[0::2] *= ratio_w
                poly[1::2] *= ratio_h
        return masks

    def resize_keyps(self, keyps, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        N, K = keyps.size()[:2]
        keyps = keyps.view(-1, 3)
        labeled = torch.nonzero(keyps[:, 2] > 0)
        if labeled.size(0) == 0:
            return keyps.view(N, K, 3)

        labeled = labeled.view(1, -1)[0]
        keyps[labeled, 0] *= ratio_w
        keyps[labeled, 1] *= ratio_h
        keyps = keyps.view(N, K, 3)
        return keyps

    def resize_semantic_seg(self, semantic_seg, scale_factor):
        ratio_h, ratio_w = _pair(scale_factor)
        return cv2.resize(
            semantic_seg, None, None,
            fx=ratio_w,
            fy=ratio_h,
            interpolation=cv2.INTER_NEAREST)


@AUGMENTATION_REGISTRY.register('keep_ar_resize')
class KeepAspectRatioResize(Resize):

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        short = min(img_w, img_h)
        large = max(img_w, img_h)
        scale = np.random.choice(self.scales)
        if scale <= 0:
            scale_factor = 1.0
        else:
            scale_factor = min(scale / short, self.max_size / large)
        return scale_factor, scale_factor


@AUGMENTATION_REGISTRY.register('keep_ar_resize_max')
class KeepAspectRatioResizeMax(Resize):
    def __init__(self,
                 max_size=sys.maxsize,
                 separate_wh=False,
                 min_size=1,
                 padding_type=None,
                 padding_val=0,
                 random_size=[],
                 scale_step=32):
        """
        Args:
            scales: evaluable str, e.g. range(500,600,1) or given list
            max_size: int
        """
        super(KeepAspectRatioResizeMax,
              self).__init__([100], max_size, separate_wh)
        self.min_size = min_size
        self.padding_type = padding_type
        self.padding_val = padding_val
        self.random_size = random_size
        self.scale_step = scale_step

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        large = max(img_w, img_h)
        if len(self.random_size) == 0:
            self.cur_size = self.max_size
        else:
            self.cur_size = random.randint(*self.random_size) * self.scale_step
        scale_factor = float(self.cur_size) / large
        return scale_factor, scale_factor

    def padding(self, data):
        image = data['image']
        height, width = image.shape[:2]
        if height == self.cur_size and width == self.cur_size:
            return data
        if image.ndim == 3:
            padding_img = np.full((self.cur_size, self.cur_size, 3),
                                  self.padding_val,
                                  dtype=image.dtype)
        else:
            padding_img = np.full((self.cur_size, self.cur_size),
                                  self.padding_val,
                                  dtype=image.dtype)
        if self.padding_type == 'left_top':
            padding_img[:height, :width] = image
        else:
            raise NotImplementedError
        data['image'] = padding_img
        return data

    def augment(self, data, scale_factor=None):
        if scale_factor is None:
            # img_h, img_w = data.image.shape[:2]
            img_h, img_w = get_image_size(data.image)
            scale_factor = self.get_scale_factor(img_h, img_w)
        output = copy.copy(data)
        output.image = self.resize_image(data.image, scale_factor)
        if self.padding_type is not None:
            output = self.padding(output)
        if self.separate_wh:
            h, w = get_image_size(data.image)
            o_h, o_w = get_image_size(output.image)
            scale_factor = (o_h / h, o_w / w)
        if has_gt_bboxes(data):
            # filter
            f_bboxes = self.resize_boxes(data.gt_bboxes, scale_factor)
            w_mask_b = (f_bboxes[:, 2] - f_bboxes[:, 0]) > self.min_size
            h_mask_b = (f_bboxes[:, 3] - f_bboxes[:, 1]) > self.min_size
            mask_b = w_mask_b & h_mask_b
            f_bboxes = f_bboxes[mask_b]
            output.gt_bboxes = f_bboxes
        if has_gt_ignores(data):
            output.gt_ignores = self.resize_boxes(data.gt_ignores,
                                                  scale_factor)
        if has_gt_keyps(data):
            output.gt_keyps = self.resize_keyps(data.gt_keyps, scale_factor)
        if has_gt_masks(data):
            output.gt_masks = self.resize_masks(data.gt_masks, scale_factor)
        if has_gt_semantic_seg(data):
            output.gt_semantic_seg = self.resize_semantic_seg(
                data.gt_semantic_seg, scale_factor)
        output.scale_factor = scale_factor
        return output

@AUGMENTATION_REGISTRY.register('copypaste_resize')
class CopyPasteResize(Resize):
    def __init__(self,
                 max_size=sys.maxsize,
                 separate_wh=False,
                 img_scale=[1024, 1024],
                 ratio_range=[0.8, 1.25]):
        """
        Args:
            scales: evaluable str, e.g. range(500,600,1) or given list
            max_size: int
        """
        super(CopyPasteResize, self).__init__([100], max_size, separate_wh)
        self.img_scale = img_scale
        self.ratio_range = ratio_range

    def get_scale_factor(self, img_h, img_w):
        """return scale_factor_h, scale_factor_w
        """
        large = max(img_w, img_h)
        min_ratio, max_ratio = self.ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(self.img_scale[0] * ratio)
        scale_factor = scale / large
        return scale_factor, scale_factor

    def resize_masks(self, masks, scale_factor):
        """
        Actually resize masks

        Note:
            Since resizing polygons will cause a shift for masks generation
            here add an offset to eliminate the difference between resizing polygons and resizing masks
        """
        ratio_h, ratio_w = _pair(scale_factor)
        masks = masks[0]
        rescaled_masks = np.stack([
            cv2.resize(
                mask, None, None,
                fx=ratio_w,
                fy=ratio_h,
                interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ])
        return [rescaled_masks]


@BATCHING_REGISTRY.register('batch_pad')
class BatchPad(object):
    def __init__(self, alignment=1, pad_value=0):
        self.alignment = alignment
        self.pad_value = pad_value

    def __call__(self, data):
        """
        Args:
            images: list of tensor
        """
        images = data['image']
        max_img_h = max([_.size(-2) for _ in images])
        max_img_w = max([_.size(-1) for _ in images])
        target_h = int(np.ceil(max_img_h / self.alignment) * self.alignment)
        target_w = int(np.ceil(max_img_w / self.alignment) * self.alignment)
        padded_images = []
        for image in images:
            assert image.dim() == 3
            src_h, src_w = image.size()[-2:]
            pad_size = (0, target_w - src_w, 0, target_h - src_h)
            padded_images.append(F.pad(image, pad_size, 'constant', self.pad_value).data)
        data['image'] = torch.stack(padded_images)
        return data

@AUGMENTATION_REGISTRY.register('copypaste_pad')
class CopyPastePad(Augmentation):
    """ random crop image base gt and crop region (iof)
    """

    def __init__(self, size=None):
        self.size = size

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = 0
        padding = (0, 0, self.size[1] - results.image.shape[1], self.size[0] - results.image.shape[0])
        padded_img = cv2.copyMakeBorder(
            results.image,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            cv2.BORDER_CONSTANT,
            value=pad_val)

        results.image = padded_img

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = 0
        masks = results.gt_masks[0]
        padding = (0, 0, self.size[1] - masks.shape[2], self.size[0] - masks.shape[1])
        if len(masks) == 0:
            padded_masks = np.empty((0, *self.size), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                cv2.copyMakeBorder(
                    mask,
                    padding[1],
                    padding[3],
                    padding[0],
                    padding[2],
                    cv2.BORDER_CONSTANT,
                    value=pad_val)
                for mask in masks
            ])
        results.gt_masks = [padded_masks]

    def augment(self, data):
        results = copy.copy(data)
        self._pad_img(results)
        self._pad_masks(results)
        return results


@AUGMENTATION_REGISTRY.register('expand')
class ImageExpand(Augmentation):
    """ expand image with means
    """

    def __init__(self, means=127.5, expand_ratios=2., expand_prob=0.5):
        self.means = means
        self.expand_ratios = expand_ratios
        self.expand_prob = expand_prob

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"
        if not coin_tossing(self.expand_prob):
            return data
        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        if check_fake_gt(new_gts):
            return data
        height, width = image.shape[:2]

        scale = random.uniform(1, self.expand_ratios)
        w = int(scale * width)
        h = int(scale * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = new_gts.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:4] += (left, top)
        if len(image.shape) == 3:
            expand_image = np.empty((h, w, 3), dtype=image.dtype)
        else:
            expand_image = np.empty((h, w), dtype=image.dtype)
        expand_image[:, :] = self.means
        expand_image[top:top + height, left:left + width] = image
        gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        output.image = expand_image
        return output


@AUGMENTATION_REGISTRY.register('fix_output_resize')
class FixOutputResize(Resize):
    """ fixed output resize
    """

    def get_scale_factor(self, img_h, img_w):
        out_h, out_w = self.scales[0], self.scales[1]
        return out_h / img_h, out_w / img_w


@AUGMENTATION_REGISTRY.register('color_jitter')
class RandomColorJitter(Augmentation):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Arguments:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        brightness_delta (int or tuple of int (min, max)): How much to jitter brightness by delta value.
            brightness_delta is chosen uniformly from [max(0, 1 - brightness_delta),
            min(0, 1 + brightness_delta)].
        hue_delta (int or tuple of int (min, max)): How much to jitter hue by delta value.
            hue_delta is chosen uniformly from [max(0, 1 - hue_delta),
            min(0, 1 + hue_delta)].
        channel_shuffle (bool): Whether to randomly shuffle the image channels or not.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, brightness_delta=0,
                 hue_delta=0, channel_shuffle=False, prob=0):
        super(RandomColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.brightness_delta = self._check_input(brightness_delta, 'brightness_delta', center=0)
        self.hue_delta = self._check_input(hue_delta, 'hue_delta', center=0)
        self.channel_shuffle = channel_shuffle
        self.prob = prob

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self, brightness, contrast, saturation, hue, brightness_delta, hue_delta, channel_shuffle):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        rETurns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        img_transforms = []

        if brightness is not None and random.random() < self.prob:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast is not None and random.random() < self.prob:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation is not None and random.random() < self.prob:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue is not None and random.random() < self.prob:
            hue_factor = random.uniform(hue[0], hue[1])
            img_transforms.append(transforms.Lambda(lambda img: adjust_hue(img, hue_factor)))

        if brightness_delta is not None and random.random() < self.prob:
            brightness_delta = random.uniform(brightness_delta[0], brightness_delta[1])
            img_transforms.append(transforms.Lambda(lambda img: img + brightness_delta))

        if hue_delta is not None and random.random() < self.prob:
            hue_delta = random.uniform(hue_delta[0], hue_delta[1])

            def augment_hue_delta(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[..., 0] += hue_delta
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                return img
            img_transforms.append(augment_hue_delta)

        if channel_shuffle and random.random() < self.prob:
            img_transforms.append(transforms.Lambda(lambda img: img[:, np.random.permutation(3)]))

        random.shuffle(img_transforms)
        img_transforms = transforms.Compose(img_transforms)

        return img_transforms

    def augment(self, data):
        """
        Arguments:
            img (np.array): Input image.
        Returns:
            img (np.array): Color jittered image.
        """
        output = copy.copy(data)
        img = data.image
        assert isinstance(img, np.ndarray)
        img = Image.fromarray(np.uint8(img))
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue,
                                    self.brightness_delta, self.hue_delta,
                                    self.channel_shuffle)
        img = transform(img)
        img = np.asanyarray(img)
        output.image = img
        return output

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


@AUGMENTATION_REGISTRY.register('crop')
class ImageCrop(Augmentation):
    """ random crop image base gt and crop region (iof)
    """

    def __init__(self, means=127.5, scale=600, crop_prob=0.5):
        self.means = means
        self.img_scale = scale
        self.crop_prob = crop_prob

    def _nobbox_crop(self, width, height, image):
        h = w = min(width, height)
        if width == w:
            left = 0
        else:
            left = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((left, t, left + w, t + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        return image_t

    def _random_crop(self, width, height, image, new_gts):
        for _ in range(100):
            if not coin_tossing(self.crop_prob):
                scale = 1
            else:
                scale = random.uniform(0.5, 1.)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                left = 0
            else:
                left = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((left, t, left + w, t + h))

            value = np_bbox_iof_overlaps(new_gts, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (new_gts[:, :2] + new_gts[:, 2:4]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = new_gts[mask].copy()

            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + ALIGNED_FLAG.offset) / w * self.img_scale
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + ALIGNED_FLAG.offset) / h * self.img_scale
            mask_b = np.minimum(b_w_t, b_h_t) >= 6.0

            if boxes_t.shape[0] == 0 or mask_b.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:])
            boxes_t[:, 2:4] -= roi[:2]
            return image_t, boxes_t
        long_side = max(width, height)
        if len(image.shape) == 3:
            image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        else:
            image_t = np.empty((long_side, long_side), dtype=image.dtype)
        image_t[:, :] = self.means
        image_t[0:0 + height, 0:0 + width] = image
        return image_t, new_gts

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        height, width = image.shape[:2]
        if check_fake_gt(new_gts):
            output.image = self._nobbox_crop(width, height, image)
            return output
        crop_image, boxes_t = self._random_crop(width, height, image, new_gts)
        gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
        output.gt_bboxes = gt_bboxes
        output.gt_ignores = ig_bboxes
        output.image = crop_image
        return output

@AUGMENTATION_REGISTRY.register('copypaste_crop')
class CopyPasteCrop(Augmentation):
    """ random crop image base gt and crop region (iof)
    """

    def __init__(self, crop_size, bbox_clip_border=True):
        self.crop_size = crop_size
        self.bbox_clip_border = bbox_clip_border

    def _crop_data(self, results, crop_size):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0

        img = results.image
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results.image = img

        # crop bboxes accordingly and clip to the image boundary
        # e.g. gt_bboxes and gt_bboxes_ignore
        bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h, 0],
                                   dtype=np.float32)
        bboxes = results.gt_bboxes.numpy() - bbox_offset
        if self.bbox_clip_border:
            bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, img_shape[1])
            bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, img_shape[0])
        valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1])
        # valid_inds = np.array([0] * len(bboxes), dtype=np.bool_)

        results.gt_bboxes = torch.tensor(bboxes[valid_inds, :], dtype=torch.float32)

        # mask fields, e.g. gt_masks and gt_masks_ignore
        masks = results.gt_masks[0][valid_inds.nonzero()[0]]
        if len(masks) == 0:
            results.gt_masks = [np.empty((0, crop_y2-crop_y1, crop_x2-crop_x1), dtype=np.uint8)]
        else:
            results.gt_masks = [masks[:, crop_y1:crop_y2, crop_x1:crop_x2]]

        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        return (min(self.crop_size[0], h), min(self.crop_size[1], w))

    def augment(self, data):
        output = copy.copy(data)
        image_size = output.image.shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(output, crop_size)
        return results


@AUGMENTATION_REGISTRY.register('stitch_expand')
class ImageStitchExpand(Augmentation):
    """ expand image with 4 original image
    """

    def __init__(self, expand_ratios=2., expand_prob=0.5):
        self.expand_ratios = expand_ratios
        self.expand_prob = expand_prob

    def augment(self, data):
        assert not has_gt_keyps(data), "key points is not supported !!!"
        assert not has_gt_masks(data), "masks is not supported !!!"

        if not coin_tossing(self.expand_prob):
            return data
        scale = random.uniform(1, self.expand_ratios)
        output = copy.copy(data)
        image = output.image
        new_gts = tensor2numpy(output)
        if check_fake_gt(new_gts) or (scale - 1) <= 1e-2:
            return data
        im_h, im_w = image.shape[:2]
        w = int(scale * im_w)
        h = int(scale * im_h)

        new_gts_tmps = []
        for j in range(2):
            for i in range(2):
                new_gts_tmp = new_gts.copy()
                new_gts_tmp[:, [0, 2]] += im_w * i
                new_gts_tmp[:, [1, 3]] += im_h * j
                new_gts_tmps.append(new_gts_tmp)
        new_gts_cat = np.concatenate(new_gts_tmps)
        height, width, = 2 * im_h, 2 * im_w
        for _ in range(10):
            ll = 0 if width == w else random.randrange(width - w)
            t = 0 if height == h else random.randrange(height - h)
            roi = np.array((ll, t, ll + w, t + h))
            value = np_bbox_iof_overlaps(new_gts_cat, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (new_gts_cat[:, :2] + new_gts_cat[:, 2:4]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = new_gts_cat[mask].copy()

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], roi[2:])
            boxes_t[:, 2:4] -= roi[:2]

            image00 = image[roi[1]:im_h, roi[0]:im_w]
            image01 = image[roi[1]:im_h, 0:roi[2] - im_w]
            image10 = image[0:roi[3] - im_h, roi[0]:im_w]
            image11 = image[0:roi[3] - im_h, 0:roi[2] - im_w]

            image_t = np.concatenate((np.concatenate((image00, image01), axis=1),
                                      np.concatenate((image10, image11), axis=1)))  # noqa
            gt_bboxes, ig_bboxes = numpy2tensor(boxes_t)
            output.gt_bboxes = gt_bboxes
            output.gt_ignores = ig_bboxes
            output.image = image_t
            return output
        return output

@AUGMENTATION_REGISTRY.register('mixup')
class MixUp(Augmentation):
    def __init__(self,
                 input_size,
                 mixup_scale,
                 extra_input=True,
                 dataset=None,
                 flip_prob=1,
                 fill_color=0,
                 clip_box=True):
        super(MixUp, self).__init__()
        assert extra_input and dataset is not None
        self.dataset = dataset
        self.input_dim = input_size
        self.mixup_scale = mixup_scale
        self.flip_prob = flip_prob
        self.fill_color = fill_color
        self.clip_box = clip_box

    def augment(self, data):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > self.flip_prob

        # data
        origin_input = copy.copy(data)
        origin_img = data['image']
        origin_labels = np.array(data['gt_bboxes'])

        # cp data
        idx_other = np.random.randint(0, len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        img = data_other['image']
        cp_labels = np.array(data_other['gt_bboxes'])

        channels = 3           # default is rgb
        if len(data.image.shape) == 2:
            channels = 1
        if not isinstance(self.fill_color, list):
            self.fill_color = [self.fill_color] * channels
        cp_img = []
        for c in range(channels):
            canvas_c = np.full((self.input_dim[0], self.input_dim[1]), self.fill_color[c], dtype=np.uint8)
            cp_img.append(canvas_c)
        if channels == 1:
            cp_img = cp_img[0]
        else:
            cp_img = np.stack(cp_img, axis=-1)
        cp_scale_ratio = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])

        # resize cp img
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        # flip cp img
        if FLIP:
            if len(data.image.shape) == 2:
                cp_img = cp_img[:, ::-1]
            else:
                cp_img = cp_img[:, ::-1, :]

        # pad
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        if len(data.image.shape) == 2:
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w))
            ).astype(np.uint8)
        else:
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w), 3)
            ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
            bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
            bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
            if self.clip_box:
                bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, w_max)
                bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, h_max)
            return bbox

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        if self.clip_box:
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2], 0, target_w
            )
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2], 0, target_h
            )

        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

            origin_input['image'] = origin_img.astype(np.uint8)
            origin_input['gt_bboxes'] = torch.tensor(origin_labels)

        return origin_input

@AUGMENTATION_REGISTRY.register('instaboost')
class InstaBoost(Augmentation):

    def __init__(
            self,
            action_candidate=('normal', 'horizontal', 'skip'),
            action_prob=(1, 0, 0),
            scale=(0.8, 1.2),
            dx=15,
            dy=15,
            theta=(-1, 1),
            color_prob=0.5,
            hflag=False,
            aug_ratio=0.5):
        super(InstaBoost, self).__init__()
        try:
            import up.tasks.det.data.datasets.instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')
        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

    def _load_anns(self, results):
        labels = results['gt_bboxes'][:, 4].numpy().astype(np.int64)
        masks = results['gt_masks']
        bboxes = results['gt_bboxes'][:, :4].numpy()
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        if len(gt_bboxes) == 0:
            results['gt_bboxes'] = torch.empty(0, 5)
            results['gt_masks'] = []
            results['image'] = img
        else:
            results['gt_bboxes'] = torch.tensor(np.hstack((gt_bboxes, gt_labels[..., np.newaxis])), dtype=torch.float32)
            results['gt_masks'] = gt_masks_ann
            results['image'] = img
        return results

    def augment(self, data):

        img = data['image']
        ori_type = img.dtype
        anns = self._load_anns(data)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            try:
                import up.tasks.det.data.datasets.instaboostfast as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            anns, img = instaboost.get_new_data(
                anns, img.astype(np.uint8), self.cfg, background=None)

        results = self._parse_anns(data, anns, img.astype(ori_type))
        del results['gt_masks']
        del results.gt_masks
        return results

@AUGMENTATION_REGISTRY.register('copypaste')
class CopyPaste(Augmentation):

    def __init__(
            self,
            extra_input=True,
            dataset=None,
            max_num_pasted=100,
            selected=True,
            transformer=None):
        super(CopyPaste, self).__init__()
        assert extra_input and dataset is not None
        self.dataset = dataset
        self.max_num_pasted = max_num_pasted
        self.selected = selected
        self.transformer = build_transformer(transformer)

    def get_idx(self):
        return random.randint(0, len(self.dataset) - 1)

    def augment(self, data):
        """Call function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        origin_input = copy.copy(data)
        origin_input['gt_masks'] = np.stack(
            [self._poly2mask(poly, origin_input['image'].shape[0], origin_input['image'].shape[1]) for poly in
             origin_input['gt_masks']])
        origin_input['gt_masks'] = [origin_input['gt_masks']]
        origin_input = self.transformer(origin_input)
        origin_input['gt_masks'] = origin_input['gt_masks'][0]

        idx = self.get_idx()
        copy_input = self.dataset.get_input(idx)
        copy_input['gt_masks'] = np.stack(
            [self._poly2mask(poly, copy_input['image'].shape[0], copy_input['image'].shape[1]) for poly in copy_input['gt_masks']])
        copy_input['gt_masks'] = [copy_input['gt_masks']]
        copy_input = self.transformer(copy_input)
        copy_input['gt_masks'] = copy_input['gt_masks'][0]

        if self.selected:
            selected_results = self._select_object(copy_input)
        else:
            selected_results = copy_input
        output = self._copy_paste(origin_input, selected_results)
        del output['gt_masks']
        del output.gt_masks

        return output

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_masks'] = selected_masks
        return results

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['image']
        dst_bboxes = dst_results['gt_bboxes'][:,:4]
        dst_labels = dst_results['gt_bboxes'][:, 4]
        dst_masks = dst_results['gt_masks']

        src_img = src_results['image']
        src_bboxes = src_results['gt_bboxes'][:,:4]
        src_labels = src_results['gt_bboxes'][:, 4]
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = self.get_bboxes(updated_dst_masks)
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes.numpy())) <= 10,
            axis=-1)
        masks_inds = updated_dst_masks.sum(
            axis=(1, 2)) > 300
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])

        dst_results['image'] = img.astype('uint8')
        dst_results['gt_bboxes'] = torch.tensor(np.concatenate((bboxes, labels[..., np.newaxis]), axis=-1))

        return dst_results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks = np.where(composed_mask, 0, masks)
        return masks

    def get_bboxes(self, masks):
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = masks.any(axis=1)
        y_any = masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes

@AUGMENTATION_REGISTRY.register('copymove')
class CopyMove(Augmentation):

    def __init__(
            self,
            aug_p = 1,
            copy_mode='hard',
            rand_scale=True,
            scale_range=[0.8,1.2],
            param=[1, 5, 2]
    ):
        super(CopyMove, self).__init__()
        self.aug_p = aug_p
        assert copy_mode in ['hard', 'soft', 'random', 'max'], 'Only hard, soft, random, max are supported!'
        self.copy_mode = copy_mode
        rc = np.load('rc.npy', allow_pickle=True).item()
        rc = torch.tensor(list(rc.values())[:1203])
        self.rc = rc
        self.rand_scale = rand_scale
        self.scale_range = scale_range
        self.param = param


    def augment(self, data):
        """Call function to make Copy-Move data augmentation.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with Copy-Move transformed.
        """
        if not coin_tossing(self.aug_p):
            return data

        origin_input = copy.copy(data)
        origin_input['gt_masks'] = np.stack(
            [self._poly2mask(poly, origin_input['image'].shape[0], origin_input['image'].shape[1]) for poly in
             origin_input['gt_masks']])

        times_result = self._copy_times(origin_input)

        output = self._copy_move(times_result)
        del output['gt_masks']
        del output.gt_masks

        return output

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _copy_times(self, results):
        """
           calculate the copy times of each instance using the rarity r_cg
        """
        gt_labels = results['gt_bboxes'][:, 4]
        r_cg = self.rc.index_select(dim=0, index=gt_labels.long() - 1)
        copy_times = torch.zeros(gt_labels.shape, dtype=torch.int64)
        if self.copy_mode == 'hard':
            copy_times[r_cg > 3] = self.param[0]
        if self.copy_mode == 'soft':
            rc_factors = torch.div(self.param[0], (1 + torch.exp((r_cg - self.param[2]) * -self.param[1])))
            copy_times = torch.round(rc_factors).long()
        if self.copy_mode == 'random':
            rc_factors = torch.div(self.param[0], (1 + torch.exp((r_cg - self.param[2]) * -self.param[1])))
            fake_copy_times = torch.round(rc_factors).long()
            fake_copy_index = torch.nonzero(fake_copy_times > 0)
            if fake_copy_index.shape[0] > 0:
                rand_index = fake_copy_index[torch.randint(fake_copy_index.shape[0], (1,))]
                copy_times[rand_index] = fake_copy_times[rand_index]
        if self.copy_mode == 'max':
            max_index = torch.argmax(r_cg)
            if r_cg[max_index] > 1:
                rc_factors = torch.div(self.param[0], (1 + torch.exp((r_cg[max_index] - self.param[2]) * -self.param[1])))
                copy_times[max_index] = torch.round(rc_factors).long()

        results['copy_times'] = copy_times
        return results

    def _copy_move(self, results):
        """Copy-Move transform function.
                Args:
                    results (dict): Result dict of the image.
                Returns:
                    dict: Updated results dict.
        """
        gt_img = results['image']
        gt_bboxes = results['gt_bboxes'][:, :4]
        gt_labels = results['gt_bboxes'][:, 4]
        gt_masks = results['gt_masks']
        gt_copy_times = results['copy_times']

        moved_masks = []
        moved_labels = []
        moved_images = []
        # Determine if instances need to be copied
        to_copy_indexes = torch.nonzero(gt_copy_times > 0)
        if to_copy_indexes.shape[0] == 0:
            return results
        # calculate Copy-Move data augmentation cyclically.
        for to_copy_index in to_copy_indexes:
            to_copy_times = gt_copy_times[to_copy_index]
            to_copy_label = gt_labels[to_copy_index]
            to_copy_bbox = gt_bboxes[to_copy_index][0]
            to_copy_mask = gt_masks[to_copy_index]
            to_copy_img = gt_img * to_copy_mask[..., np.newaxis]
            for i in range(to_copy_times):
                moved_mask = np.zeros((to_copy_mask.shape[0], to_copy_mask.shape[1]), dtype=np.uint8)
                moved_image = np.zeros((to_copy_mask.shape[0], to_copy_mask.shape[1], 3), dtype=np.uint8)
                rand_h = np.random.randint(to_copy_mask.shape[0])
                rand_w = np.random.randint(to_copy_mask.shape[1])
                if self.rand_scale == True:
                    sc_factor = np.random.uniform(self.scale_range[0],self.scale_range[1])
                else:
                    sc_factor = 1
                resized_to_copy_mask = cv2.resize(
                    to_copy_mask, None, None,
                    fx=sc_factor,
                    fy=sc_factor,
                    interpolation=cv2.INTER_LINEAR)
                resized_to_copy_bbox = to_copy_bbox * sc_factor
                resized_to_copy_img = cv2.resize(
                    to_copy_img, None, None,
                    fx=sc_factor,
                    fy=sc_factor,
                    interpolation=cv2.INTER_LINEAR)
                move_need_h = moved_mask.shape[0] - rand_h
                move_need_w = moved_mask.shape[1] - rand_w
                copy_have_h = resized_to_copy_mask.shape[0] - int(resized_to_copy_bbox[1])
                copy_have_w = resized_to_copy_mask.shape[1] - int(resized_to_copy_bbox[0])
                pad_h = 0
                pad_w = 0
                if copy_have_h < move_need_h:
                    pad_h = move_need_h - copy_have_h
                if copy_have_w < move_need_w:
                    pad_w = move_need_w - copy_have_w
                pad_resized_to_copy_mask = np.pad(resized_to_copy_mask, ((0, pad_h), (0, pad_w)), 'constant')
                pad_resized_to_copy_img = np.pad(resized_to_copy_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                moved_mask[rand_h:,rand_w:] = pad_resized_to_copy_mask[int(resized_to_copy_bbox[1]):int(resized_to_copy_bbox[1])+move_need_h,
                                              int(resized_to_copy_bbox[0]):int(resized_to_copy_bbox[0])+move_need_w]
                moved_image[rand_h:, rand_w:] = pad_resized_to_copy_img[
                                               int(resized_to_copy_bbox[1]):int(resized_to_copy_bbox[1]) + move_need_h,
                                               int(resized_to_copy_bbox[0]):int(resized_to_copy_bbox[0]) + move_need_w]
                moved_masks.append(moved_mask)
                moved_labels.append(to_copy_label)
                moved_images.append(moved_image)
        # deal with the occlusion between the copied instances.
        for i in range(len(moved_masks)-2, -1, -1):
            union_mask = np.where(np.any(np.stack(moved_masks[i+1:]), axis=0), 1, 0)
            moved_masks[i] = np.where(union_mask, 0, moved_masks[i])
            moved_images[i] = moved_images[i] * moved_masks[i][..., np.newaxis]

        # handle the occlusion of the original instances by the copied instance.
        moved_masks = np.stack(moved_masks)
        composed_mask = np.where(np.any(moved_masks, axis=0), 1, 0)
        updated_gt_masks = self.get_updated_masks(gt_masks, composed_mask)
        updated_gt_bboxes = self.get_bboxes(updated_gt_masks)
        moved_bboxes = self.get_bboxes(moved_masks)
        assert len(updated_gt_masks) == len(updated_gt_bboxes)
        bboxes_inds = np.all(
            np.abs(
                (updated_gt_bboxes - gt_bboxes.numpy())) <= 10,
            axis=-1)
        masks_inds = updated_gt_masks.sum(
            axis=(1, 2)) > 300
        valid_inds = bboxes_inds | masks_inds

        # Copy and move objects in image
        img = gt_img * (1 - composed_mask[..., np.newaxis])
        for i in range(len(moved_images)):
            img = img + moved_images[i]
        bboxes = np.concatenate([updated_gt_bboxes[valid_inds], moved_bboxes])
        labels = torch.cat((gt_labels[valid_inds], torch.stack(moved_labels).squeeze(-1)))

        results['image'] = img.astype('uint8')
        results['gt_bboxes'] = torch.tensor(np.concatenate((bboxes, labels[..., np.newaxis]), axis=-1))

        return results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks = np.where(composed_mask, 0, masks)
        return masks

    def get_bboxes(self, masks):
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = masks.any(axis=1)
        y_any = masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes

@AUGMENTATION_REGISTRY.register('rand_copymove')
class randCopyMove(Augmentation):

    def __init__(
            self,
            aug_p = 1,
            rand_scale=True,
            scale_range=[0.8,1.2],
            copytimes_cate=5
    ):
        super(randCopyMove, self).__init__()
        self.aug_p = aug_p
        self.rand_scale = rand_scale
        self.scale_range = scale_range
        self.copytimes_cate = copytimes_cate
        self.now_copy_times = np.zeros(1203, dtype=np.int64)


    def augment(self, data):

        if not coin_tossing(self.aug_p):
            return data

        origin_input = copy.copy(data)
        origin_input['gt_masks'] = np.stack(
            [self._poly2mask(poly, origin_input['image'].shape[0], origin_input['image'].shape[1]) for poly in
             origin_input['gt_masks']])

        times_result = self._copy_times(origin_input)

        output = self._copy_move(times_result)
        del output['gt_masks']
        del output.gt_masks

        return output

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _copy_times(self, results):
        """
            rand Copy-Move copies copytimes_cate instances for each class.
        """
        gt_labels = results['gt_bboxes'][:, 4]
        copy_times = torch.zeros(gt_labels.shape, dtype=torch.int64)
        unique_labels = torch.unique(gt_labels-1)
        for unique_label in unique_labels:
            if self.now_copy_times[int(unique_label.item())] != self.copytimes_cate:
                first_index = torch.nonzero(gt_labels == (unique_label + 1))[0]
                copy_times[first_index] = self.copytimes_cate
                self.now_copy_times[int(unique_label.item())] = self.copytimes_cate


        results['copy_times'] = copy_times
        return results

    def _copy_move(self, results):
        """Copy-Move transform function.
                Args:
                    results (dict): Result dict of the image.
                Returns:
                    dict: Updated results dict.
        """
        gt_img = results['image']
        gt_bboxes = results['gt_bboxes'][:, :4]
        gt_labels = results['gt_bboxes'][:, 4]
        gt_masks = results['gt_masks']
        gt_copy_times = results['copy_times']

        moved_masks = []
        moved_labels = []
        moved_images = []
        # Determine if instances need to be copied
        to_copy_indexes = torch.nonzero(gt_copy_times > 0)
        if to_copy_indexes.shape[0] == 0:
            return results
        # calculate Copy-Move data augmentation cyclically.
        for to_copy_index in to_copy_indexes:
            to_copy_times = gt_copy_times[to_copy_index]
            to_copy_label = gt_labels[to_copy_index]
            to_copy_bbox = gt_bboxes[to_copy_index][0]
            to_copy_mask = gt_masks[to_copy_index]
            to_copy_img = gt_img * to_copy_mask[..., np.newaxis]
            for i in range(to_copy_times):
                moved_mask = np.zeros((to_copy_mask.shape[0], to_copy_mask.shape[1]), dtype=np.uint8)
                moved_image = np.zeros((to_copy_mask.shape[0], to_copy_mask.shape[1], 3), dtype=np.uint8)
                rand_h = np.random.randint(to_copy_mask.shape[0])
                rand_w = np.random.randint(to_copy_mask.shape[1])
                if self.rand_scale == True:
                    sc_factor = np.random.uniform(self.scale_range[0],self.scale_range[1])
                else:
                    sc_factor = 1
                resized_to_copy_mask = cv2.resize(
                    to_copy_mask, None, None,
                    fx=sc_factor,
                    fy=sc_factor,
                    interpolation=cv2.INTER_LINEAR)
                resized_to_copy_bbox = to_copy_bbox * sc_factor
                resized_to_copy_img = cv2.resize(
                    to_copy_img, None, None,
                    fx=sc_factor,
                    fy=sc_factor,
                    interpolation=cv2.INTER_LINEAR)
                move_need_h = moved_mask.shape[0] - rand_h
                move_need_w = moved_mask.shape[1] - rand_w
                copy_have_h = resized_to_copy_mask.shape[0] - int(resized_to_copy_bbox[1])
                copy_have_w = resized_to_copy_mask.shape[1] - int(resized_to_copy_bbox[0])
                pad_h = 0
                pad_w = 0
                if copy_have_h < move_need_h:
                    pad_h = move_need_h - copy_have_h
                if copy_have_w < move_need_w:
                    pad_w = move_need_w - copy_have_w
                pad_resized_to_copy_mask = np.pad(resized_to_copy_mask, ((0, pad_h), (0, pad_w)), 'constant')
                pad_resized_to_copy_img = np.pad(resized_to_copy_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                moved_mask[rand_h:,rand_w:] = pad_resized_to_copy_mask[int(resized_to_copy_bbox[1]):int(resized_to_copy_bbox[1])+move_need_h,
                                              int(resized_to_copy_bbox[0]):int(resized_to_copy_bbox[0])+move_need_w]
                moved_image[rand_h:, rand_w:] = pad_resized_to_copy_img[
                                               int(resized_to_copy_bbox[1]):int(resized_to_copy_bbox[1]) + move_need_h,
                                               int(resized_to_copy_bbox[0]):int(resized_to_copy_bbox[0]) + move_need_w]
                moved_masks.append(moved_mask)
                moved_labels.append(to_copy_label)
                moved_images.append(moved_image)
        # deal with the occlusion between the copied instances.
        for i in range(len(moved_masks)-2, -1, -1):
            union_mask = np.where(np.any(np.stack(moved_masks[i+1:]), axis=0), 1, 0)
            moved_masks[i] = np.where(union_mask, 0, moved_masks[i])
            moved_images[i] = moved_images[i] * moved_masks[i][..., np.newaxis]

        # handle the occlusion of the original instances by the copied instance.
        moved_masks = np.stack(moved_masks)
        composed_mask = np.where(np.any(moved_masks, axis=0), 1, 0)
        updated_gt_masks = self.get_updated_masks(gt_masks, composed_mask)
        updated_gt_bboxes = self.get_bboxes(updated_gt_masks)
        moved_bboxes = self.get_bboxes(moved_masks)
        assert len(updated_gt_masks) == len(updated_gt_bboxes)
        bboxes_inds = np.all(
            np.abs(
                (updated_gt_bboxes - gt_bboxes.numpy())) <= 10,
            axis=-1)
        masks_inds = updated_gt_masks.sum(
            axis=(1, 2)) > 300
        valid_inds = bboxes_inds | masks_inds

        # Copy and move objects in image
        img = gt_img * (1 - composed_mask[..., np.newaxis])
        for i in range(len(moved_images)):
            img = img + moved_images[i]
        bboxes = np.concatenate([updated_gt_bboxes[valid_inds], moved_bboxes])
        labels = torch.cat((gt_labels[valid_inds], torch.stack(moved_labels).squeeze(-1)))

        results['image'] = img.astype('uint8')
        results['gt_bboxes'] = torch.tensor(np.concatenate((bboxes, labels[..., np.newaxis]), axis=-1))

        return results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks = np.where(composed_mask, 0, masks)
        return masks

    def get_bboxes(self, masks):
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = masks.any(axis=1)
        y_any = masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes

@AUGMENTATION_REGISTRY.register('statis_instance')
class StatisInstance(Augmentation):
    """
       count the number of all instances augmented during 1 epoch
    """

    def __init__(self, savename):
        super(StatisInstance, self).__init__()
        self.savename = savename
        self.num_instance = np.zeros(1203, dtype=np.int64)

    def augment(self, data):

        output = copy.copy(data)
        labels = output.gt_bboxes[:, 4]
        d = Counter(labels.long().cpu().numpy() - 1)
        lb_keys = list(d.keys())
        lb_values = list(d.values())
        self.num_instance[lb_keys] = self.num_instance[lb_keys] + lb_values
        np.save(self.savename+'.npy', self.num_instance)
        return output