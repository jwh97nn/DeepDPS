import numbers
from PIL import Image

import numpy as np
from detectron2.data.transforms import Augmentation, ResizeTransform
from fvcore.transforms.transform import Transform
from torchvision.transforms import functional as F


class ResizeTransformWithCamMatrixAug(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.

    ResizeShortestEdgeWithCamMatrixAug uses the same get_transform() as in ResizeShortestEdge
    (see https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py),  # noqa
    but uses ResizeTransformWithCamMatrixAug instead of ResizeTransform
    """

    def __init__(self, new_h, new_w, interp=Image.ANTIALIAS):
        super().__init__()
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def get_transform(self, image):
        h, w = image.shape[:2]
        return ResizeTransformWithCamMatrixTrans(
            h, w, self.new_h, self.new_w, self.interp
        )


class ResizeTransformWithCamMatrixTrans(ResizeTransform):
    def apply_coords(self, coords):
        # apply_coords is used for optical center transformation
        coords[:, 0] = (coords[:, 0] + 0.5) * (self.new_w * 1.0 / self.w) - 0.5
        coords[:, 1] = (coords[:, 1] + 0.5) * (self.new_h * 1.0 / self.h) - 0.5
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_focal(self, coords):
        # apply_focal uses default resize scaling to augment focal lengths
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class ColorJitterAug(Augmentation):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_transform(self, image):
        fn_idx = np.random.permutation(4)

        b = (
            None
            if self.brightness is None
            else float(np.random.uniform(self.brightness[0], self.brightness[1]))
        )
        c = (
            None
            if self.contrast is None
            else float(np.random.uniform(self.contrast[0], self.contrast[1]))
        )
        s = (
            None
            if self.saturation is None
            else float(np.random.uniform(self.saturation[0], self.saturation[1]))
        )
        h = (
            None
            if self.hue is None
            else float(np.random.uniform(self.hue[0], self.hue[1]))
        )

        return ColorJitterTransform(fn_idx, b, c, s, h)


class ColorJitterTransform(Transform):
    def __init__(
        self,
        fn_idx,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor,
    ):
        super().__init__()
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        img = Image.fromarray(img)

        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = F.adjust_brightness(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = F.adjust_contrast(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = F.adjust_saturation(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = F.adjust_hue(img, self.hue_factor)

        return np.asarray(img)

    def inverse(self) -> Transform:
        raise NotImplementedError()
