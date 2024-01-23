# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""

logger = logging.getLogger(__name__)


def load_cityscapes_dvps_panoptic(image_dir, gt_dir, gt_json, meta):
    assert os.path.exists(gt_json), f"{gt_json} does not exist"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    ret = []
    for file_dict in file_dicts:
        ret.append(
            {
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(file_dict["image"]))[0].split(
                        "_"
                    )[:3]
                ),
                "height": file_dict["height"],
                "width": file_dict["width"],
                "file_name": os.path.join(image_dir, file_dict["image"]),
                "vps_label_file_name": os.path.join(image_dir, file_dict["seg"]),
                "depth_label_file_name": os.path.join(image_dir, file_dict["depth"]),
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    return ret


_RAW_CITYSCAPES_DPS_SPLITS = {
    "cityscapes_dvps_train": (
        "Cityscapes-DVPS/video_sequence/train",
        "Cityscapes-DVPS/video_sequence/train",
        "Cityscapes-DVPS/video_sequence/dvps_cityscapes_train.json",
    ),
    "cityscapes_dvps_val": (
        "Cityscapes-DVPS/video_sequence/val",
        "Cityscapes-DVPS/video_sequence/val",
        "Cityscapes-DVPS/video_sequence/dvps_cityscapes_val.json",
    ),
}


def register_all_cityscapes_dvps(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_CITYSCAPES_DPS_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key,
            lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes_dvps_panoptic(
                x, y, z, meta
            ),
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_dvps",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


if __name__ == "__main__":

    # root = "/home/junwen/Datasets/"
    root = "/hjw/Datasets/"
    register_all_cityscapes_dvps(root)
    dataset = DatasetCatalog.get("cityscapes_dvps_val")
    meta = MetadataCatalog.get("cityscapes_dvps_val")
    # print(dataset[-1])
    print(len(dataset))
    # print(dataset[-1].keys())
    print(meta)
