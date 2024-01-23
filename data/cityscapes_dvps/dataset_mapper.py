# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

try:
    from .transform import ColorAugSSDTransform
except:
    from transform import ColorAugSSDTransform

__all__ = ["CityscapesDVPSDatasetMapper"]


class CityscapesDVPSDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        semantic_guided,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.semantic_guided = semantic_guided

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = []
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "semantic_guided": cfg.MODEL.MASK_FORMER.SEMANTIC_GUIDED,
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
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "vps_label_file_name" in dataset_dict:
            vps_seg_gt = utils.read_image(
                dataset_dict.pop("vps_label_file_name")
            )  # int32
            # sem_seg_gt = vps_seg_gt // 1000
            # sem_seg_gt[sem_seg_gt > 18] = self.ignore_label
            # sem_seg_gt = sem_seg_gt.astype(np.uint8)
            # vps_seg_gt, convert_dict = dense_ind(
            #     vps_seg_gt, stay_shape=True, shuffle=False, return_convert=True
            # )
            # print(np.unique(vps_seg_gt))
            # print(convert_dict)

            # print(vps_seg_gt.dtype)
            # pan_seg_gt = np.ones_like(vps_seg_gt) * 255
            # pan_seg_gt[vps_seg_gt <= 10] = vps_seg_gt[vps_seg_gt <= 10]
            # print(np.unique(pan_seg_gt))
            # print(np.unique(vps_seg_gt))

            vps_seg_gt, mapping = dense_id(vps_seg_gt)
            vps_seg_gt = vps_seg_gt.astype(np.uint8)
            # print(np.unique(vps_seg_gt))
            # print(mapping)

        else:
            # vps_seg_gt, sem_seg_gt = None, None
            vps_seg_gt = None

        if "depth_label_file_name" in dataset_dict:
            depth_gt = utils.read_image(
                dataset_dict.pop("depth_label_file_name")
            )  # int32
            depth_gt_1 = (depth_gt // 256).astype(np.uint8)
            depth_gt_2 = (depth_gt % 256).astype(np.uint8)
        else:
            depth_gt = None

        aug_input = T.AugInput(image, sem_seg=vps_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        if vps_seg_gt is not None:
            vps_seg_gt = aug_input.sem_seg
            vps_seg_gt = vps_seg_gt.astype(np.int32)
            for k, v in mapping.items():
                vps_seg_gt[vps_seg_gt == k] = v
            # print(np.unique(vps_seg_gt))

        if depth_gt is not None:
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(
                np.float64
            )
            depth_gt = depth_gt / 256.0
            del depth_gt_1, depth_gt_2

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if vps_seg_gt is not None:
            vps_seg_gt = torch.as_tensor(vps_seg_gt.astype("long"))
        if depth_gt is not None:
            dataset_dict["depth"] = torch.as_tensor(
                np.ascontiguousarray(depth_gt).astype("float32")
            )

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        pan_seg_gt = vps_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        for id in np.unique(pan_seg_gt):
            mask = pan_seg_gt == id
            # h, w = mask.shape
            # num_pixel = h * w
            # if mask.sum() < 0.005 * num_pixel or id == 255:
            if mask.sum() < 100 or id == 255:
                continue

            if id <= 10:  # Stuff
                classes.append(id)
                # masks.append(pan_seg_gt == id)
            elif id != 255:  # Thing
                classes.append(id // 1000)
                # masks.append(pan_seg_gt == id)
            masks.append(mask)

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
                )
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        if self.semantic_guided:
            dataset_dict["sem_seg"] = vps_seg_gt

        return dataset_dict


def dense_id(vps_seg_gt, start_id=11):
    """keep 0-10 and 255
    others: #class*1000+id -> (from 11 to 11+#instance)
    """
    unique_ids = np.unique(vps_seg_gt)
    mapping = {}
    for i in range(10 + 1):
        mapping[i] = i
    mapping[255] = 255
    for id in unique_ids:
        if id > 10 and id != 255:
            vps_seg_gt[vps_seg_gt == id] = start_id
            mapping[start_id] = id
            start_id += 1
    return vps_seg_gt, mapping


if __name__ == "__main__":
    from detectron2.data import (
        build_detection_train_loader,
        build_detection_test_loader,
    )
    from detectron2.config import get_cfg

    from detectron2.data import DatasetCatalog
    import sys
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from cityscapes_dvps import register_all_cityscapes_dvps
    import cv2

    # sys.path.append("/home/junwen/Depth/shit-d/")
    sys.path.append("/hjw/Depth/shit-d/")
    from maskformer_config import add_maskformer2_config

    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file("/hjw/Depth/shit-d/configs/maskformer/test.yaml")
    cfg.freeze()

    register_all_cityscapes_dvps(root="/hjw/Datasets/")
    mapper = CityscapesDVPSDatasetMapper(cfg, is_train=True)
    d = build_detection_train_loader(cfg, mapper=mapper)
    # d = build_detection_test_loader(cfg, )
    for batched_inputs in d:
        print(f"keys: {batched_inputs[0].keys()}")

        gt_instances = batched_inputs[0]["instances"]
        print(gt_instances.gt_classes)
        gt_masks = gt_instances.gt_masks
        print([m.sum() for m in gt_masks])
        assert 0
