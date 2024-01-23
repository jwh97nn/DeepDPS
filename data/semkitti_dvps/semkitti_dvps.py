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

# fmt: off
SEMKITTI_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 1, "id": 10, "trainId": 0, "name": "car"},
    {"color": (244, 35, 232), "isthing": 1, "id": 11, "trainId": 1, "name": "bicycle"},
    {"color": (70, 70, 70), "isthing": 1, "id": 15, "trainId": 2, "name": "motorcycle"},
    {"color": (102, 102, 156), "isthing": 1, "id": 18, "trainId": 3, "name": "truck"},
    {"color": (190, 153, 153), "isthing": 1, "id": 20, "trainId": 4, "name": "other-vehicle"},
    {"color": (153, 153, 153), "isthing": 1, "id": 30, "trainId": 5, "name": "person"},
    {"color": (250, 170, 30), "isthing": 1, "id": 31, "trainId": 6, "name": "bicyclist"},
    {"color": (220, 220, 0), "isthing": 1, "id": 32, "trainId": 7, "name": "motorcyclist"},
    {"color": (107, 142, 35), "isthing": 0, "id": 40, "trainId": 8, "name": "road"},
    {"color": (152, 251, 152), "isthing": 0, "id": 44, "trainId": 9, "name": "parking"},
    {"color": (70, 130, 180), "isthing": 0, "id": 48, "trainId": 10, "name": "sidewalk"},
    {"color": (220, 20, 60), "isthing": 0, "id": 49, "trainId": 11, "name": "other-ground"},
    {"color": (255, 0, 0), "isthing": 0, "id": 50, "trainId": 12, "name": "building"},
    {"color": (0, 0, 142), "isthing": 0, "id": 51, "trainId": 13, "name": "fence"},
    {"color": (0, 0, 70), "isthing": 0, "id": 70, "trainId": 14, "name": "vegetation"},
    {"color": (0, 60, 100), "isthing": 0, "id": 71, "trainId": 15, "name": "trunk"},
    {"color": (0, 80, 100), "isthing": 0, "id": 72, "trainId": 16, "name": "terrain"},
    {"color": (0, 0, 230), "isthing": 0, "id": 80, "trainId": 17, "name": "pole"},
    {"color": (119, 11, 32), "isthing": 0, "id": 81, "trainId": 18, "name": "traffic-sign"},
]
[ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
KITTI_CITYSCAPE = {
    0: 13,
    1: 18,
    2: 17,
    3: 14,
    4: 15,
    5: 11,
    6: 12,
    7: 16,  # train

    8: 0,
    9: 10, # sky
    10: 1,
    11: 3,  # wall
    12: 2,
    13: 4,
    14: 8,
    15: 6,  # traffic light
    16: 9,
    17: 5,
    18: 7,
}
CITYSCAPE_KITTI = {v: k for k, v in KITTI_CITYSCAPE.items()}

KITTI_MV = {
    0: 56,  # car
    1: 53,  # bicycle
    2: 58,  # motorcycle
    3: 62,  # truck
    4: 60,  # other-vehicle
    5: 20,  # person
    6: 21,  # bicyclist
    7: 22,  # motorcyclist
    
    8: 14,  # road
    9: 11,  # parking
    10: 16,  # sidewalk
    11: 8,  # other-ground (9,10,12,13,15,)
    12: 18,  # building
    13: 4,  # fence
    14: 31,  # vegetation
    15: 26,  # trunk
    16: 30,  # terrain
    17: 48,  # pole
    18: 47,  # traffic-sign (50, 51)
}

MV_KITTI = {v: k for k, v in KITTI_MV.items()}
for i in [55, 57, 59, 61]:  # object--vehicle--bus, object--vehicle--caravan, object--vehicle--on-rails, object--vehicle--trailer
    MV_KITTI[i] = 4  
for i in [24, 25, 44]:  # marking--crosswalk-zebra, marking--general, object--pothole
    MV_KITTI[i] = 8  
for i in [9, 10, 12, 13, 15]:  # other-ground
    MV_KITTI[i] = 11
for i in [3, 10]:   # construction--barrier--curb, construction--flat--curb-cut
    MV_KITTI[i] = 10 
for i in [46]:  # object--support--pole
    MV_KITTI[i] = 17
for i in [49, 50, 51]:  # object--traffic-light, object--traffic-sign--back, object--traffic-sign--front, 
    MV_KITTI[i] = 18
# fmt: on

import numpy as np


def transfer_kitti_to_cityscapes(id):
    max_ins = 1000
    new_map = np.ones_like(id) * 255 * max_ins
    for curr_id in np.unique(id):
        k_id = curr_id // max_ins
        if k_id == 255:
            continue
        c_id = KITTI_CITYSCAPE[k_id]
        if k_id <= 7:  # Things
            ins_id = curr_id % max_ins
            new_id = c_id * max_ins + ins_id
        else:
            new_id = c_id * max_ins
        new_map[id == curr_id] = new_id
    return new_map


def transfer_kitti_to_mv(id):
    max_ins = 1000
    new_map = np.ones_like(id) * 255 * max_ins
    for curr_id in np.unique(id):
        k_id = curr_id // max_ins
        if k_id == 255:
            continue
        c_id = KITTI_MV[k_id]
        if k_id <= 7:  # Things
            ins_id = curr_id % max_ins
            new_id = c_id * max_ins + ins_id
        else:
            new_id = c_id * max_ins
        new_map[id == curr_id] = new_id
    return new_map


def load_semkitti_dvps_panoptic(image_dir, gt_dir, gt_json, meta):
    assert os.path.exists(gt_json), f"{gt_json} does not exist"
    with open(gt_json) as f:
        file_dicts = json.load(f)

    if "train" in gt_json:
        # replace = 'selected'  # 2643
        replace = "selected_1_7_11"  # 3590
        with open(gt_json.replace("dvps_kitti_train", replace)) as f:
            selected = json.load(f)
        is_train = True
    else:
        is_train = False

    ret = []
    for file_dict in file_dicts:
        if is_train:
            if file_dict["image"] not in selected:
                continue

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

    # return ret
    return ret[:1]


# {
# 'image_id': '000000_000000_leftImg8bit',
# 'height': 376,
# 'width': 1241,
# 'file_name': '/hjw/Datasets/SemKITTI-DVPS/video_sequence/train/000000_000000_leftImg8bit.png',
# 'vps_label_file_name': '/hjw/Datasets/SemKITTI-DVPS/video_sequence/train/000000_000000_gtFine_class.png',
# 'depth_label_file_name': '/hjw/Datasets/SemKITTI-DVPS/video_sequence/train/000000_000000_depth_718.8560180664062.png'
# }


_RAW_SEMKITTI_DPS_SPLITS = {
    "semkitti_dvps_train": (
        "SemKITTI-DVPS/video_sequence/train",
        "SemKITTI-DVPS/video_sequence/train",
        "SemKITTI-DVPS/video_sequence/dvps_kitti_train.json",
    ),
    # "semkitti_dvps_val": (
    #     "SemKITTI-DVPS/video_sequence/val",
    #     "SemKITTI-DVPS/video_sequence/val",
    #     "SemKITTI-DVPS/video_sequence/dvps_kitti_val.json",
    # ),
    "semkitti_dvps_val": (
        "SemKITTI-DVPS/video_sequence/train",
        "SemKITTI-DVPS/video_sequence/train",
        "SemKITTI-DVPS/video_sequence/dvps_kitti_train.json",
    ),
}


def register_all_semkitti_dvps(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in SEMKITTI_CATEGORIES]
    thing_colors = [k["color"] for k in SEMKITTI_CATEGORIES]
    stuff_classes = [k["name"] for k in SEMKITTI_CATEGORIES]
    stuff_colors = [k["color"] for k in SEMKITTI_CATEGORIES]

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

    for k in SEMKITTI_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_SEMKITTI_DPS_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        DatasetCatalog.register(
            key,
            lambda x=image_dir, y=gt_dir, z=gt_json: load_semkitti_dvps_panoptic(
                x, y, z, meta
            ),
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir,
            evaluator_type="semkitti_dvps",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


if __name__ == "__main__":

    # root = "/home/junwen/Datasets/"
    root = "/hjw/Datasets/"
    register_all_semkitti_dvps(root)
    # dataset = DatasetCatalog.get("semkitti_dvps_val")
    # meta = MetadataCatalog.get("semkitti_dvps_val")
    dataset = DatasetCatalog.get("semkitti_dvps_train")
    # meta = MetadataCatalog.get("semkitti_dvps_train")

    print(len(dataset))
    # print(dataset[-1].keys())
    # print(meta)
