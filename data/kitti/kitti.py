import json
import os
from typing import Dict, List

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

IMAGE_FOLDER = {
    "left": "image_02",
    "right": "image_03",
}
CALIB_FILE = {
    "cam2cam": "calib_cam_to_cam.txt",
    "velo2cam": "calib_velo_to_cam.txt",
    "imu2velo": "calib_imu_to_velo.txt",
}

_RAW_KITTI_EIGEN_SPLITS = {
    "kitti_zhou_train": ("data_splits/eigen_zhou_files.txt", ""),
    "kitti_eigen_test_original": ("data_splits/eigen_test_files.txt", "original"),
    "kitti_eigen_test_improved": ("data_splits/eigen_test_files.txt", "improved"),
}


def load_kitti_eigen(root: str, image_split_file: str, depth_type: str) -> List[dict]:
    """
    Args:
       root (str): path to dataset root.
       image_split_file (str): path to the data split file.
       gt_dir (str): path to the raw annotations.
           e.g., "~/kitti_eigen/gtFine_sequence/kitti_eigen_panoptic_train".
       meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id" and
           "stuff_dataset_id_to_contiguous_id" to map category ids to contiguous ids for training.

    Returns:
       list[dict]: a list of dicts in Detectron2 standard format. (See
           `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    calibration_cache = {}

    with open(os.path.join(root, image_split_file)) as f:
        files = f.read().splitlines()
    files = [x.split(" ")[0] for x in files]

    ret = []
    for file in files:
        image_file = os.path.join(root, file)

        # Get previous and next frame for current image_file from the video sequence dir
        image_idx = int(image_file.split("/")[-1][:-4])
        image_prev_file = (
            image_file[:-14] + str(image_idx - 1).zfill(10) + image_file[-4:]
        )
        image_next_file = (
            image_file[:-14] + str(image_idx + 1).zfill(10) + image_file[-4:]
        )

        # Skip first and last samples in video sequence in train set
        if "zhou" in image_split_file and (
            not os.path.exists(image_prev_file) or not os.path.exists(image_next_file)
        ):
            continue

        if "test" in image_split_file:
            depth_file = _get_depth_file(image_file, depth_type)

            # Skip test sample if depth file is not available
            if not os.path.exists(depth_file):
                continue
        else:
            depth_file = None

        # Add intrinsics
        parent_folder = _get_parent_folder(image_file)
        if parent_folder in calibration_cache:
            c_data = calibration_cache[parent_folder]
        else:
            c_data = _read_raw_calib_file(parent_folder)
            calibration_cache[parent_folder] = c_data
        intrinsics = _get_intrinsics(image_file, c_data)

        # Convert to Cityscapes format
        calibration_info = dict()
        calibration_info["intrinsic"] = dict()
        calibration_info["intrinsic"]["fx"] = intrinsics[0][0]
        calibration_info["intrinsic"]["fy"] = intrinsics[1][1]
        calibration_info["intrinsic"]["u0"] = intrinsics[0][2]
        calibration_info["intrinsic"]["v0"] = intrinsics[1][2]
        # calibration_info["intrinsic"]["fx"] = 720.36
        # calibration_info["intrinsic"]["fy"] = 621
        # calibration_info["intrinsic"]["u0"] = 720
        # calibration_info["intrinsic"]["v0"] = 187.5

        ret.append(
            {
                "file_name": image_file,
                "image_id": file[:-4],
                "depth_file_name": depth_file,
                "prev_img_file_name": image_prev_file,
                "next_img_file_name": image_next_file,
                "calibration_info": calibration_info,
            }
        )
    assert len(ret), f"No images found from data split file {image_split_file}!"

    return ret


def register_kitti_eigen(root):
    for key, (image_split_file, depth_type) in _RAW_KITTI_EIGEN_SPLITS.items():
        DatasetCatalog.register(
            key,
            lambda r=root, i=image_split_file, d=depth_type: load_kitti_eigen(r, i, d),
        )

        MetadataCatalog.get(key).set(
            image_split_file=image_split_file, evaluator_type="kitti_eigen"
        )


def _get_parent_folder(image_file):
    """Get the parent folder from image_file."""
    return os.path.abspath(os.path.join(image_file, "../../../.."))


def _get_depth_file(image_file, depth_type):
    """Get the corresponding depth file from an image file."""
    if depth_type == "original":
        folder = "velodyne"
        suffix = "npz"
    elif depth_type == "improved":
        folder = "groundtruth"
        suffix = "png"

    for cam in ["left", "right"]:
        if IMAGE_FOLDER[cam] in image_file:
            depth_file = image_file.replace(
                IMAGE_FOLDER[cam] + "/data",
                "proj_depth/{}/{}".format(folder, IMAGE_FOLDER[cam]),
            )
            depth_file = depth_file[:-3] + suffix

            return depth_file


def _get_intrinsics(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary."""
    for cam in ["left", "right"]:
        # Check for both cameras, if found replace and return intrinsics
        if IMAGE_FOLDER[cam] in image_file:
            return np.reshape(
                calib_data[IMAGE_FOLDER[cam].replace("image", "P_rect")], (3, 4)
            )[:, :3]


def _read_raw_calib_file(folder):
    """Read raw calibration files from folder."""
    filepath = os.path.join(folder, CALIB_FILE["cam2cam"])

    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
