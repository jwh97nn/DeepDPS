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

try:
    from .semkitti_dvps import (
        KITTI_CITYSCAPE,
        CITYSCAPE_KITTI,
        KITTI_MV,
        transfer_kitti_to_cityscapes,
        transfer_kitti_to_mv,
    )
except:
    from semkitti_dvps import (
        KITTI_CITYSCAPE,
        CITYSCAPE_KITTI,
        KITTI_MV,
        transfer_kitti_to_cityscapes,
        transfer_kitti_to_mv,
    )

__all__ = ["SemKITTIDVPSDatasetMapper"]


class SemKITTIDVPSDatasetMapper:
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
        transfer_cityscapes=False,
        transfer_mv=False,
        train_depth=True,
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
        self.transfer_cityscapes = transfer_cityscapes
        self.transfer_mv = transfer_mv
        self.train_depth = train_depth

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
                    interp=Image.NEAREST,
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
            # augs = [
            #     T.Resize(
            #         (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
            #         interp=Image.NEAREST,
            #     ),
            # ]
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())
        else:
            augs = [
                T.Resize(
                    (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
                    interp=Image.BILINEAR,
                ),
            ]

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
            "transfer_cityscapes": cfg.MODEL.MASK_FORMER.TRANSFER_CITYSCAPES,
            "transfer_mv": cfg.MODEL.MASK_FORMER.TRANSFER_MV,
            "train_depth": cfg.MODEL.MASK_FORMER.TRAIN_DEPTH,
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
            # label_file_name = dataset_dict.pop("vps_label_file_name")
            label_file_name = dataset_dict["vps_label_file_name"]
            vps_seg_gt_class = utils.read_image(label_file_name)
            vps_seg_gt_instance = utils.read_image(
                label_file_name.replace("class", "instance")
            )

            # 0-7: Things, 8-18: Stuff
            # gt_class: 0-18, 255: ignore
            # gt_instance: 1-.., 0: ignore
            max_ins = 1000
            vps_seg_gt = vps_seg_gt_class * max_ins + vps_seg_gt_instance  # int32

            vps_seg_gt, mapping = dense_id(vps_seg_gt)
            vps_seg_gt = vps_seg_gt.astype(np.uint8)
        else:
            vps_seg_gt = None

        if "depth_label_file_name" in dataset_dict and self.train_depth:
            # depth_gt = utils.read_image(
            #     dataset_dict.pop("depth_label_file_name")
            # )  # int32
            depth_gt = utils.read_image(dataset_dict["depth_label_file_name"])  # int32
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
        # np.save("image.npy", image)
        # np.save("vps_seg_gt.npy", vps_seg_gt)

        if depth_gt is not None:
            depth_gt_1 = transforms.apply_segmentation(depth_gt_1)
            depth_gt_2 = transforms.apply_segmentation(depth_gt_2)
            depth_gt = depth_gt_1.astype(np.float64) * 256 + depth_gt_2.astype(
                np.float64
            )
            depth_gt = depth_gt / 256.0
            # print(depth_gt.max())
            del depth_gt_1, depth_gt_2
        # np.save("depth.npy", depth_gt)
        # assert 0

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

        if self.transfer_cityscapes:  # Transfer SemKITTI id to Cityscapes id
            pan_seg_gt = transfer_kitti_to_cityscapes(pan_seg_gt)
        elif self.transfer_mv:
            pan_seg_gt = transfer_kitti_to_mv(pan_seg_gt)
            thing_ids, stuff_ids = [], []
            for i in range(8):  # Things
                thing_ids.append(KITTI_MV[i])
            for i in range(8, 19):  # Stuff
                stuff_ids.append(KITTI_MV[i])

        instances = Instances(image_shape)
        classes = []
        masks = []
        for id in np.unique(pan_seg_gt):
            mask = pan_seg_gt == id
            # h, w = mask.shape
            # num_pixel = h * w
            # if mask.sum() < 0.005 * num_pixel or id == 255:
            if id == 255 * max_ins:
                continue
            # if mask.sum() <= 10 or id == 255 * max_ins:
            #     continue

            class_id = id // max_ins
            if self.transfer_cityscapes:
                # 11-18: Things, 0-10: Stuff
                class_id_in_stuff = class_id <= 10
            elif self.transfer_mv:
                class_id_in_stuff = class_id in stuff_ids
            else:
                # 0-7: Things, 8-18: Stuff
                # gt_class: 0-18, 255: ignore
                # gt_instance: 1-.., 0: ignore
                class_id_in_stuff = class_id >= 8 and class_id <= 18

            if class_id_in_stuff:
                classes.append(class_id)
                # masks.append(pan_seg_gt == id)
            elif class_id != 255:
                if id % max_ins == 0:
                    continue
                classes.append(class_id)
                # masks.append(pan_seg_gt == id)
            masks.append(mask)

        valid_mask = pan_seg_gt != 255 * max_ins
        dataset_dict["valid_mask"] = torch.as_tensor(valid_mask)

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


def dense_id(vps_seg_gt, max_ins=1000, start_id=19):
    """keep 8-18 and 255
    others: #class*1000+id -> (from 19 to 19+#instance)
    """
    unique_ids = np.unique(vps_seg_gt)
    mapping = {}
    for id in unique_ids:
        class_id = id // max_ins
        if (class_id >= 8 and class_id <= 18) or class_id == 255:
            mapping[class_id] = id
            vps_seg_gt[vps_seg_gt == id] = class_id
        else:
            instance_id = id % max_ins
            if instance_id == 0:  # ignore
                vps_seg_gt[vps_seg_gt == id] = 255
            else:  # real instance
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
    from semkitti_dvps import register_all_semkitti_dvps
    import cv2

    # sys.path.append("/home/junwen/Depth/shit-d/")
    sys.path.append("/hjw/Depth/shit-d/")
    from maskformer_config import add_maskformer2_config

    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file("/hjw/Depth/shit-d/configs/maskformer/r50_semkitti.yaml")
    cfg.freeze()

    register_all_semkitti_dvps(root="/hjw/Datasets/")
    # register_all_kitti_dvps(root="/home/junwen/Datasets/")
    mapper = SemKITTIDVPSDatasetMapper(cfg, is_train=True)
    d = build_detection_train_loader(cfg, mapper=mapper)
    # d = build_detection_test_loader(cfg, )

    gt_classes = torch.zeros(19, dtype=torch.int64)

    for i, batched_inputs in enumerate(d):
        # print(f"keys: {batched_inputs[0].keys()}")

        for b in batched_inputs:
            gt_instances = b["instances"]
            # print(gt_instances.gt_classes)
            # assert 0

            for cls in gt_instances.gt_classes:
                gt_classes[CITYSCAPE_KITTI[int(cls)]] += 1
            # assert 0

        if i % 100 == 0:
            print(gt_classes)
    print(gt_classes)

    # for i in range(5):
    #     gt_instances = batched_inputs[i]["instances"]
    #     print(gt_instances.gt_classes)
    #     gt_masks = gt_instances.gt_masks
    #     print([m.sum() for m in gt_masks])

    #     img = batched_inputs[i]["image"].numpy()
    #     np.save(f"img{i}.npy", img)

    #     gt_masks = gt_masks.numpy()
    #     np.save(f"gt_masks{i}.npy", gt_masks)

    #     gt_instances = gt_instances.gt_classes.numpy()
    #     np.save(f"gt_instances{i}.npy", gt_instances)

    # depth = batched_inputs[0]["depth"].numpy()
    # np.save("depth.npy", depth)

    # assert 0


# Val
# [
# 0: 16012,
# 1: 886,
# 2: 339,
# 3: 127,
# 4: 1021,
# 5: 1375,
# 6: 713,
# 7: 65,
# 8: 4071,
# 9: 21,
# 10: 4016,
# 11: 158,
# 12: 3606,
# 13: 2530,
# 14: 4071,
# 15: 3243,
# 16: 3632,
# 17: 3740,
# 18: 1680
# ]
