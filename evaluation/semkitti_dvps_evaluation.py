# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

try:
    from . import eval_dvps_semkitti as semkitti_eval
except ImportError:
    import eval_dvps_semkitti as semkitti_eval


class SemKITTIDPSEvaluation(CityscapesEvaluator):
    """
    Evaluate video panoptic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def __init__(self, dataset_name, output_folder, depth_thres=[-1, 0.5, 0.25, 0.1]):
        super().__init__(dataset_name)
        self.output_folder = output_folder
        self.evaluator = semkitti_eval
        # self.eval_frames = [1, 2, 3, 4]
        self.depth_thres = depth_thres
        # self.depth_thres = [-1]

    def process(self, inputs, outputs):
        save_dir = self._temp_dir
        for input, output in zip(inputs, outputs):
            gt_name = os.path.basename(input["vps_label_file_name"])
            gt_depth_name = os.path.basename(input["depth_label_file_name"])

            # root = "/hjw/Datasets/SemKITTI-DVPS/video_sequence/val/"
            if output["depth"] is not None:
                pred_depth = output["depth"].to(self._cpu_device).numpy()
                # pred_depth = (
                #     np.array(Image.open(os.path.join(root, gt_depth_name))).astype(
                #         np.float32
                #     )
                #     / 256.0
                # )
                pred_depth = (pred_depth * 256).astype(np.int32)
                Image.fromarray(pred_depth).save(os.path.join(save_dir, gt_depth_name))

            # result_panoptic = output["panoptic_seg_dvps"]  # [H, W]
            # result_panoptic = result_panoptic.cpu().numpy().astype(np.uint32)

            result_panoptic_class = output["panoptic_seg_dvps_class"]
            result_panoptic_instance = output["panoptic_seg_dvps_instance"]
            result_panoptic_class = (
                result_panoptic_class.cpu().numpy().astype(np.uint32)
            )
            result_panoptic_instance = (
                result_panoptic_instance.cpu().numpy().astype(np.uint32)
            )

            # result_panoptic_class = np.array(
            #     Image.open(os.path.join(root, gt_name))
            # ).astype(np.uint32)
            # result_panoptic_instance = np.array(
            #     Image.open(os.path.join(root, gt_name.replace("class", "instance")))
            # ).astype(np.uint32)

            Image.fromarray(result_panoptic_class).save(os.path.join(save_dir, gt_name))
            Image.fromarray(result_panoptic_instance).save(
                os.path.join(save_dir, gt_name.replace("class", "instance"))
            )

            # Image.fromarray(pred_depth).save(os.path.join(save_dir, gt_depth_name))

            Image.fromarray(result_panoptic_class).save(
                os.path.join(
                    "output/semkitti/1/a/",
                    gt_name,
                )
            )
            Image.fromarray(result_panoptic_instance).save(
                os.path.join(
                    "output/semkitti/1/a/",
                    gt_name.replace("class", "instance"),
                )
            )
            # Image.fromarray(result_panoptic_instance).save(
            #     os.path.join(
            #         "output/city-dvps/58/depth",
            #         basename.replace("_leftImg8bit.png", "_depth.png"),
            #     )
            # )

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        return self.evaluate_dpq()

    def evaluate_dpq(self):
        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        dpq = {}
        pred_dir = self._temp_dir
        gt_dir = self._metadata.gt_dir

        ret = OrderedDict()

        if len(self.depth_thres) > 1:  # [-1] = only PQ
            depth_errs = self.evaluate_depth()
            ret["depth"] = depth_errs

        for depth_thres in self.depth_thres:
            results = self.evaluator.main(1, pred_dir, gt_dir, depth_thres)
            dpq[str(depth_thres)] = {
                "dpq": results["averages"][0],
                "dpq_th": results["averages"][1],
                "dpq_st": results["averages"][2],
            }

        ret.update(dpq)

        if len(self.depth_thres) > 1:
            ret["averages"] = {
                "dpq": np.array(
                    [
                        dpq[str(depth_thres)]["dpq"]
                        for depth_thres in self.depth_thres
                        if depth_thres > 0
                    ]
                ).mean(),
                "dpq_th": np.array(
                    [
                        dpq[str(depth_thres)]["dpq_th"]
                        for depth_thres in self.depth_thres
                        if depth_thres > 0
                    ]
                ).mean(),
                "dpq_st": np.array(
                    [
                        dpq[str(depth_thres)]["dpq_st"]
                        for depth_thres in self.depth_thres
                        if depth_thres > 0
                    ]
                ).mean(),
            }

        self._working_dir.cleanup()
        return ret

    def evaluate_depth(self):
        # Load depth groundtruth
        gt_dir = self._metadata.gt_dir
        depth_gt_names = os.scandir(gt_dir)
        depth_gt_names = [name.name for name in depth_gt_names if "depth" in name.name]
        # depth_gts = load_sequence(depth_gt_names)
        depth_gts = sorted(depth_gt_names)
        depth_gts = [
            np.array(Image.open(os.path.join(gt_dir, name))) for name in depth_gts
        ]

        # Load depth prediction
        pred_dir = self._temp_dir
        depth_pred_names = os.scandir(pred_dir)
        depth_pred_names = [
            name.name for name in depth_pred_names if "depth" in name.name
        ]
        # depth_preds = load_sequence(depth_pred_names)
        depth_preds = sorted(depth_pred_names)
        depth_preds = [
            np.array(Image.open(os.path.join(pred_dir, name))) for name in depth_preds
        ]

        depth_preds = np.stack(depth_preds, axis=0)
        depth_gts = np.stack(depth_gts, axis=0)

        num_samples = depth_preds.shape[0]
        silog = np.zeros(num_samples, np.float32)
        log10 = np.zeros(num_samples, np.float32)
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1 = np.zeros(num_samples, np.float32)
        d2 = np.zeros(num_samples, np.float32)
        d3 = np.zeros(num_samples, np.float32)

        for i, (gt, pred) in enumerate(zip(depth_gts, depth_preds)):
            valid = depth_gts[i] > 0
            (
                silog[i],
                log10[i],
                abs_rel[i],
                sq_rel[i],
                rms[i],
                log_rms[i],
                d1[i],
                d2[i],
                d3[i],
            ) = compute_errors(gt[valid], pred[valid])

        return {
            "silog": silog.mean(),
            "abs_rel": abs_rel.mean(),
            "log10": log10.mean(),
            "rms": rms.mean(),
            "sq_rel": sq_rel.mean(),
            "log_rms": log_rms.mean(),
            "d1": d1.mean(),
            "d2": d2.mean(),
            "d3": d3.mean(),
        }


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
