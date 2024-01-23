import itertools
from collections import OrderedDict

import numpy as np
from PIL import Image
import cv2
from detectron2.data import detection_utils as utils
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.utils import comm

import sys

sys.path.append("..")
from model.loss_mono import disp_to_depth


class DepthEvaluator(CityscapesEvaluator):
    """
    Evaluate depth metrics within a distance of [min_depth, max_depth].
    Labels outside this distance are masked.

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(
        self,
        dataset_name: str,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        use_gt_scale: bool = False,
        use_eigen_crop: bool = False,
    ):
        """
        Args:
            dataset_name (str): the name of the dataset.
            min_depth (float): minimum depth in GT considered for metrics.
            max_depth (float): maximum depth in GT considered for metrics.
            use_gt_scale (bool): whether to use GT median scaling for the prediction or not.
            use_eigen_crop (bool): whether to use the commonly used eigen crop for evaluation.
        """
        super().__init__(dataset_name)
        self._errors = []
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._use_gt_scale = use_gt_scale
        self._use_eigen_crop = use_eigen_crop
        self._ratios = []
        self.dataset_name = dataset_name

    def reset(self):
        self._errors = []
        self._ratios = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # prediction = output["disps"]["disp", 0].to(self._cpu_device).numpy()
            prediction = output.to(self._cpu_device).numpy()
            prediction, _ = disp_to_depth(prediction, 0.1, 100)

            # Read gt file and convert it to metric depth
            if "depth_file_name" in input:
                label = read_depth(input["depth_file_name"]).astype(np.float32)
            elif "disparity_file_name" in input:
                label = utils.read_image(input["disparity_file_name"]).astype(
                    np.float32
                )
                # Convert saved format to disparity
                label[label != 0] = (label[label != 0] - 1.0) / 256.0
                # Convert disparity to depth
                factor = (
                    input["calibration_info"]["extrinsic"]["baseline"]
                    * input["calibration_info"]["intrinsic"]["fx"]
                )
                label[label != 0] = factor / label[label != 0]
            else:
                raise RuntimeError(
                    "Neither depth_file_name nor disparity_file_name are given for the dataset. "
                    "Impossible to run DepthEvaluator!"
                )

            h, w = label.shape
            mask = np.logical_and(label > self._min_depth, label < self._max_depth)

            if self._use_eigen_crop:
                crop = np.array(
                    [
                        0.40810811 * label.shape[-2],
                        0.99189189 * label.shape[-2],
                        0.03594771 * label.shape[-1],
                        0.96405229 * label.shape[-1],
                    ]
                ).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            prediction = cv2.resize(prediction[0], (w, h))
            prediction = 1 / prediction
            prediction = prediction[mask]
            label = label[mask]

            if self._use_gt_scale:
                ratio = np.median(label) / np.median(prediction)
                self._ratios.append(ratio)
                prediction *= ratio

            prediction[prediction < self._min_depth] = self._min_depth
            prediction[prediction > self._max_depth] = self._max_depth

            thresh = np.maximum((label / prediction), (prediction / label))
            a1 = (thresh < 1.25).mean()
            a2 = (thresh < 1.25**2).mean()
            a3 = (thresh < 1.25**3).mean()

            rmse = (label - prediction) ** 2
            rmse = np.sqrt(rmse.mean())

            rmse_log = (np.log(label) - np.log(prediction)) ** 2
            rmse_log = np.sqrt(rmse_log.mean())

            abs_rel = np.mean(np.abs(label - prediction) / label)

            sq_rel = np.mean(((label - prediction) ** 2) / label)

            self._errors.append([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

    def evaluate(self):
        comm.synchronize()
        errors = comm.gather(self._errors, dst=0)
        errors = list(itertools.chain(*errors))
        scale_ratios = comm.gather(self._ratios, dst=0)
        scale_ratios = list(itertools.chain(*scale_ratios))

        if not comm.is_main_process():
            return

        mean_errors = np.array(errors).mean(0)

        if "original" in self.dataset_name:
            depth_type = "original"
        elif "improved" in self.dataset_name:
            depth_type = "improved"

        self._logger.info(
            "{:14s}| {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}  {:>5s}".format(
                depth_type + "Depth",
                "Abs Rel",
                "Sq Rel",
                "RMSE",
                "RMSE log",
                "\u03B4 < 1.25",
                "\u03B4 < 1.25\u00b2",
                "\u03B4 < 1.25\u00b3",
            )
        )
        self._logger.info("-" * 80)
        # fmt: off
        self._logger.info(
            "{:14s}|   {:5.3f}   {:5.3f}  {:5.3f}     {:5.3f}     {:5.3f}      {:5.3f}      {:5.3f}".format(  # noqa
                "ALL", *mean_errors
            )
        )
        # fmt: on

        if self._use_gt_scale:
            self._logger.info("-" * 80)
            ratios = np.array(scale_ratios)
            median = np.median(scale_ratios)
            self._logger.info(
                "Scaling ratios | median: {:0.3f} | std: {:0.3f}".format(
                    median, np.std(ratios / median)
                )
            )

        ret = OrderedDict()
        ret[depth_type] = {
            "Abs Rel": mean_errors[0],
            "Sq Rel": mean_errors[1],
            "RMSE": mean_errors[2],
            "RMSE log": mean_errors[3],
            "\u03B4 < 1.25": mean_errors[4],
            "\u03B4 < 1.25\u00b2": mean_errors[5],
            "\u03B4 < 1.25\u00b3": mean_errors[6],
        }

        return ret


def read_depth(depth_file):
    """Get the depth map from a file."""
    if depth_file.endswith(".npz"):
        return read_npz_depth(depth_file)
    elif depth_file.endswith(".png"):
        return read_png_depth(depth_file)
    else:
        raise NotImplementedError("Depth type not implemented")


def read_npz_depth(file):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)["velodyne" + "_depth"].astype(np.float32)
    # return np.expand_dims(depth, axis=2)
    return depth


def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert np.max(depth_png) > 255, "Wrong .png depth file"
    depth = depth_png.astype(np.float) / 256.0
    depth[depth_png == 0] = -1.0
    # return np.expand_dims(depth, axis=2)
    return depth


def load_image(path):
    """
    Read an image using PIL
    Parameters
    ----------
    path : str
        Path to the image
    Returns
    -------
    image : PIL.Image
        Loaded image
    """
    return Image.open(path)
