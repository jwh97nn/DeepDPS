# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from .transform import ColorAugSSDTransform

# from transform import ColorAugSSDTransform

# from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

__all__ = ["PanopticDatasetMapper"]


class PanopticDatasetMapper:
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

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
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
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert (
            self.is_train
        ), "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # print(np.unique(sem_seg_gt))
        # print(np.unique(pan_seg_gt))
        # assert 0

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError(
                "Pemantic segmentation dataset should not have 'annotations'."
            )

        # Prepare per-category binary masks
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                # classes.append(class_id)
                # masks.append(pan_seg_gt == segment_info["id"])

                mask = pan_seg_gt == segment_info["id"]
                if mask.sum() < 100:
                    continue
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

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
        # instances.gt_masks = masks

        dataset_dict["instances"] = instances

        return dataset_dict


if __name__ == "__main__":
    from detectron2.data import build_detection_train_loader
    from detectron2.config import get_cfg

    # from detectron2.data.datasets.cityscapes_panoptic import (
    #     register_all_cityscapes_panoptic,
    # )
    from detectron2.data import DatasetCatalog
    import sys
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from cityscapes import register_all_cityscapes_panoptic
    import cv2

    sys.path.append("/home/junwen/Depth/shit-d/")
    from maskformer_config import add_maskformer2_config

    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file("/home/junwen/Depth/shit-d/configs/maskformer/r50.yaml")
    cfg.freeze()

    register_all_cityscapes_panoptic(root="/home/junwen/Datasets/")
    mapper = PanopticDatasetMapper(cfg, is_train=True)
    d = build_detection_train_loader(cfg, mapper=mapper)
    for batched_inputs in d:
        print(batched_inputs[0].keys())

        sem_seg = batched_inputs[0]["sem_seg"]
        # print(sem_seg.shape)
        # print(sem_seg.dtype)
        print(sem_seg.unique())

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        gt_instances = batched_inputs[0]["instances"]

        print(gt_instances.gt_classes)
        gt_masks = gt_instances.gt_masks
        # print(gt_masks.shape)
        print([m.sum() for m in gt_masks])

        assert 0
        # print(len(ins))
        # print(batched_inputs[0]["segments_info"])

        # visulizer.draw_panoptic_seg(
        #     batched_inputs[0]["sem_seg"], batched_inputs[0]["segments_info"]
        # )
        # visulizer.draw_sem_seg(batched_inputs[0]["sem_seg"])
        # visulizer.get_output().save("img1.png")

        # h_pad, w_pad = img.shape[:2]
        # gt_masks = gt_instances.gt_masks
        # padded_masks = torch.zeros(
        #     (gt_masks.shape[0], h_pad, w_pad),
        #     dtype=gt_masks.dtype,
        #     device=gt_masks.device,
        # )
        # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

        # for i in range(padded_masks.shape[0]):
        #     m = (padded_masks[i].float() * 255).unsqueeze(0).repeat(3, 1, 1)
        #     m = m.permute(1, 2, 0).numpy().astype(np.uint8)
        #     cv2.imwrite("mask_%d.png" % i, m)
        # print(gt_instances.gt_classes)

        assert 0
