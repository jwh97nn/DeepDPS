import copy
import logging
from typing import Callable, List, Union
import cv2
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import DatasetMapper, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from ..transform import ResizeTransformWithCamMatrixAug, ColorJitterAug


class TrainDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name".
       If depth training is enabled, it further reads image_prev and image_next
       as well as camera calibration infos from "calibration_info"
    2. Applies random scale, crop, flip and color jitter transforms to images, label, camera matrix
    3. Prepare data to Tensor and generate training targets
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        color_jitter_augmentation: T.Augmentation,
        image_format: str,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            color_jitter_augmentation: color_jitter_augmentation which is applied to the image.
                None, if no color_jitter_augmentation is used.
            image_format: an image format supported by :func:`detection_utils.read_image`.
            with_depth: whether to create targets for self-supervised depth training.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
            depth_ignore_ids: a list of category ids which will be ignored in depth training.
                Usually this includes ego vehicle and sky label.
        """
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.color_jitter_augmentation = color_jitter_augmentation
        self.image_format = image_format

        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg):
        augs = [
            ResizeTransformWithCamMatrixAug(
                new_h=cfg.INPUT.HEIGHT, new_w=cfg.INPUT.WIDTH
            )
        ]
        augs.append(T.RandomFlip())

        color_jitter_aug = ColorJitterAug(
            brightness=cfg.INPUT.COLOR_JITTER.BRIGHTNESS,
            contrast=cfg.INPUT.COLOR_JITTER.CONTRAST,
            saturation=cfg.INPUT.COLOR_JITTER.SATURATION,
            hue=cfg.INPUT.COLOR_JITTER.HUE,
        )

        ret = {
            "augmentations": augs,
            "color_jitter_augmentation": color_jitter_aug,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load images.
        image_orig = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, image_orig)
        image_prev_orig, image_next_orig = None, None

        image_prev_orig = utils.read_image(
            dataset_dict["prev_img_file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, image_prev_orig)
        image_next_orig = utils.read_image(
            dataset_dict["next_img_file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, image_next_orig)

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image_orig)
        tfl = self.augmentations(aug_input)
        image_orig = aug_input.image

        # Apply color jitter augmentation separately.
        # Original images will be used for photometric loss calculation.
        color_jitter_tf = None
        if self.color_jitter_augmentation is not None:
            color_jitter_aug_input = T.AugInput(image_orig)
            color_jitter_tf = self.color_jitter_augmentation(color_jitter_aug_input)
            image = color_jitter_aug_input.image
        else:
            image = image_orig

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose([2, 0, 1]))
        )

        # Apply image augmentations to context images
        image_prev_orig = tfl.apply_image(image_prev_orig)
        image_next_orig = tfl.apply_image(image_next_orig)

        if self.color_jitter_augmentation is not None:
            image_prev = color_jitter_tf.apply_image(image_prev_orig)
            image_next = color_jitter_tf.apply_image(image_next_orig)
        else:
            image_prev = image_prev_orig
            image_next = image_next_orig

        dataset_dict["image_orig"] = torch.as_tensor(
            np.ascontiguousarray(image_orig.transpose([2, 0, 1]))
        )
        dataset_dict["image_prev_orig"] = torch.as_tensor(
            np.ascontiguousarray(image_prev_orig.transpose([2, 0, 1]))
        )
        dataset_dict["image_prev"] = torch.as_tensor(
            np.ascontiguousarray(image_prev.transpose([2, 0, 1]))
        )
        dataset_dict["image_next_orig"] = torch.as_tensor(
            np.ascontiguousarray(image_next_orig.transpose([2, 0, 1]))
        )
        dataset_dict["image_next"] = torch.as_tensor(
            np.ascontiguousarray(image_next.transpose([2, 0, 1]))
        )

        # Generate camera matrix targets.
        optical_center = np.array(
            [
                dataset_dict["calibration_info"]["intrinsic"]["u0"],
                dataset_dict["calibration_info"]["intrinsic"]["v0"],
            ]
        ).reshape(1, 2)
        focal_length = np.array(
            [
                dataset_dict["calibration_info"]["intrinsic"]["fx"],
                dataset_dict["calibration_info"]["intrinsic"]["fy"],
            ]
        ).reshape(1, 2)

        # Apply augmentations
        # Use apply_coords() to augment optical center values.
        optical_center = tfl.apply_coords(optical_center)
        # noinspection PyTypeChecker
        for tf in tfl:
            try:
                focal_length = tf.apply_focal(focal_length)
            except AttributeError:
                pass

        # fmt: off
        camera_matrix = np.array([[focal_length[0, 0],                  0, optical_center[0, 0], 0],   # noqa
                                  [                 0, focal_length[0, 1], optical_center[0, 1], 0],   # noqa
                                  [                 0,                  0,                    1, 0],   # noqa
                                  [                 0,                  0,                    0, 1]],  # noqa
                                  dtype=np.float32)
        # fmt: on
        dataset_dict["camera_matrix"] = torch.as_tensor(
            camera_matrix.astype(np.float32)
        )

        return dataset_dict


class TestDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name".
       If depth training is enabled, it further reads image_prev and image_next
       as well as camera calibration infos from "calibration_info"
    2. Applies random scale, crop, flip and color jitter transforms to images, label, camera matrix
    3. Prepare data to Tensor and generate training targets
    """

    @configurable
    def __init__(
        self,
        is_train: bool = False,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            color_jitter_augmentation: color_jitter_augmentation which is applied to the image.
                None, if no color_jitter_augmentation is used.
            image_format: an image format supported by :func:`detection_utils.read_image`.
            with_depth: whether to create targets for self-supervised depth training.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
            depth_ignore_ids: a list of category ids which will be ignored in depth training.
                Usually this includes ego vehicle and sky label.
        """
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format

    @classmethod
    def from_config(cls, cfg):
        augs = [
            ResizeTransformWithCamMatrixAug(
                new_h=cfg.INPUT.HEIGHT, new_w=cfg.INPUT.WIDTH
            )
        ]

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load images.
        image_orig = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, image_orig)

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image_orig)
        tfl = self.augmentations(aug_input)
        image_orig = aug_input.image

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_orig.transpose([2, 0, 1]))
        )

        return dataset_dict
