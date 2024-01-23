import numpy as np
from PIL import Image
import six
import os
import multiprocessing as mp
import pdb
import sys
import time
import argparse


def vpq_eval(element):
    pred_ids, gt_ids = element
    max_ins = 2**20
    ign_id = 255
    offset = 2**30
    num_cat = 20

    # max_ins = 1000
    # offset = 256 * 256

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas) if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    # int_ids = gt_ids.astype(np.uint64) * offset + pred_ids.astype(np.uint64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)

        # Tag: ignore void
        # if pred_cat == ign_id or gt_cat == ign_id:
        if pred_cat == ign_id:
            continue

        if gt_cat != pred_cat:
            continue
        union = (
            gt_areas[gt_id]
            + pred_areas[pred_id]
            - int_area
            - prediction_void_overlap(pred_id)
        )

        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = int(gt_id // max_ins)
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = int(pred_id // max_ins)

        # # Tag: ignore void
        # if cat == ign_id:
        #     continue

        fp_per_class[cat] += 1

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)


def eval(element):
    max_ins = 2**20
    # max_ins = 1000

    pred_cat, pred_ins, gts, depth_preds, depth_gts, depth_thres = element
    pred_cat = [np.array(Image.open(image)) for image in pred_cat]
    pred_ins = [np.array(Image.open(image)) for image in pred_ins]
    pred_cat = np.concatenate(pred_cat, axis=1)
    pred_ins = np.concatenate(pred_ins, axis=1)
    pred = pred_cat.astype(np.int32) * max_ins + pred_ins.astype(np.int32)
    # preds = pred_cat.astype(np.uint64) * max_ins + pred_ins.astype(np.uint64)

    gts_cat = [np.array(Image.open(image)) for image in gts]
    gts_ins = [
        np.array(Image.open(image.replace("class", "instance"))) for image in gts
    ]
    gts = [
        gt_cat.astype(np.int32) * max_ins + gt_ins.astype(np.int32)
        # gt_cat.astype(np.uint64) * max_ins + gt_ins.astype(np.uint64)
        for gt_cat, gt_ins in zip(gts_cat, gts_ins)
    ]
    gt = np.concatenate(gts, axis=1)
    for id in np.unique(gt):
        cls_id = id // max_ins
        if cls_id <= 7:
            ins_id = id % max_ins
            if ins_id == 0:  # ignore
                gt[gt == id] = 255 * max_ins

    abs_rel = 0
    if depth_thres > 0:
        depth_preds = [np.array(Image.open(name)) for name in depth_preds]
        depth_gts = [np.array(Image.open(name)) for name in depth_gts]
        depth_preds = np.concatenate(depth_preds, axis=1)
        depth_gts = np.concatenate(depth_gts, axis=1)
        depth_mask = depth_gts > 0
        abs_rel = np.mean(
            np.abs(depth_preds[depth_mask] - depth_gts[depth_mask])
            / depth_gts[depth_mask]
        )

        pred_in_mask = pred[:, : depth_preds.shape[1]]
        pred_in_depth_mask = pred_in_mask[depth_mask]
        ignored_pred_mask = (
            np.abs(depth_preds[depth_mask] - depth_gts[depth_mask])
            / depth_gts[depth_mask]
        ) > depth_thres
        pred_in_depth_mask[ignored_pred_mask] = 19 * max_ins
        pred_in_mask[depth_mask] = pred_in_depth_mask
        pred[:, : depth_preds.shape[1]] = pred_in_mask

        # for depth_pred, depth_gt, pred in zip(depth_preds, depth_gts, preds):
        #     depth_mask = depth_gt > 0
        #     pred_in_depth_mask = pred[depth_mask]
        #     ignored_pred_mask = (
        #         np.abs(depth_pred[depth_mask] - depth_gt[depth_mask])
        #         / depth_gt[depth_mask]
        #     ) > depth_thres
        #     pred_in_depth_mask[ignored_pred_mask] = 19 * max_ins
        #     pred[depth_mask] = pred_in_depth_mask

    # result = vpq_eval([pred, gt])
    result = vpq_eval([pred, gt])

    return result + (abs_rel,)


def main(eval_frames, pred_dir, gt_dir, depth_thres):
    gt_names = os.scandir(gt_dir)
    gt_names = [name.name for name in gt_names if "gtFine_class" in name.name]
    gt_names = [os.path.join(gt_dir, name) for name in gt_names]
    gt_names = sorted(gt_names)

    depth_gt_names = os.scandir(gt_dir)
    depth_gt_names = [name.name for name in depth_gt_names if "depth" in name.name]
    depth_gt_names = [os.path.join(gt_dir, name) for name in depth_gt_names]
    depth_gt_names = sorted(depth_gt_names)

    if depth_thres > 0:
        depth_pred_names = os.scandir(pred_dir)
        depth_pred_names = [
            name.name for name in depth_pred_names if "depth" in name.name
        ]
        depth_pred_names = [os.path.join(pred_dir, name) for name in depth_pred_names]
        depth_pred_names = sorted(depth_pred_names)
    else:
        depth_pred_names = []

    pred_names = os.scandir(pred_dir)
    pred_names = [os.path.join(pred_dir, name.name) for name in pred_names]
    cat_pred_names = [name for name in pred_names if "class" in name]
    ins_pred_names = [name for name in pred_names if "instance" in name]
    cat_pred_names = sorted(cat_pred_names)
    ins_pred_names = sorted(ins_pred_names)

    # Overfit test
    # cat_pred_names = [n for n in cat_pred_names if "000000_000000" in n]
    # ins_pred_names = [n for n in ins_pred_names if "000000_000000" in n]
    # gt_names = [n for n in gt_names if "000000_000000" in n]

    all_lst = []
    for i in range(len(cat_pred_names) - eval_frames + 1):
        all_lst.append(
            [
                cat_pred_names[i : i + eval_frames],
                ins_pred_names[i : i + eval_frames],
                gt_names[i : i + eval_frames],
                depth_pred_names[i : i + eval_frames],
                depth_gt_names[i : i + eval_frames],
                depth_thres,
            ]
        )

    N = mp.cpu_count() // 2 + 1
    with mp.Pool(processes=N) as p:
        results = p.map(eval, all_lst)
    # results = [eval(lst) for lst in all_lst]

    iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)
    tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)
    fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)
    fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)
    abs_rel = np.stack([result[4] for result in results]).mean(axis=0)

    iou_per_class = iou_per_class[:19]
    tp_per_class = tp_per_class[:19]
    fn_per_class = fn_per_class[:19]
    fp_per_class = fp_per_class[:19]

    epsilon = 1e-10
    sq = iou_per_class / (tp_per_class + epsilon)
    rq = tp_per_class / (
        tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + epsilon
    )

    # epsilon = 1
    # sq = iou_per_class / np.maximum(tp_per_class, epsilon)
    # rq = tp_per_class / np.maximum(
    #     tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class, epsilon
    # )

    print("iou_per_class: ", iou_per_class)
    print("tp_per_class: ", tp_per_class)
    print("fn_per_class: ", fn_per_class)
    print("fp_per_class: ", fp_per_class)

    pq = sq * rq
    print("sq: ", sq)
    print("rq: ", rq)
    print("pq: ", pq)

    spq = pq[8:]
    tpq = pq[:8]
    # print(
    #     r"{:.1f} {:.1f} {:.1f}".format(
    #         pq.mean() * 100, tpq.mean() * 100, spq.mean() * 100
    #     )
    # )

    print(
        "k={}, lambda={}, result:\n".format(eval_frames, depth_thres),
        "PQ     PQ_th  PQ_st\n",
        "{:.2f}  {:.2f}  {:.2f}".format(
            pq.mean() * 100, tpq.mean() * 100, spq.mean() * 100
        ),
    )
    ret = dict()
    ret["averages"] = (pq.mean() * 100, tpq.mean() * 100, spq.mean() * 100)
    ret["classes"] = [(x * 100, y * 100, z * 100) for x, y, z in zip(pq, sq, rq)]
    return ret


if __name__ == "__main__":
    pred_dir = "/hjw/Depth/shit-d/output/kitti-dvps/1/"
    # pred_dir = "/hjw/Depth/shit-d/output/semkitti/1/a/"
    # pred_dir = "/hjw/Datasets/SemKITTI-DVPS/video_sequence/val/"
    gt_dir = "/hjw/Datasets/SemKITTI-DVPS/video_sequence/val/"
    # pred_dir = "/home/junwen/Depth/shit-d/output/semkitti/1/a/"
    # pred_dir = "/home/junwen/Datasets/SemKITTI-DVPS/video_sequence/val/"
    # gt_dir = "/home/junwen/Datasets/SemKITTI-DVPS/video_sequence/val/"
    # pred_dir = "/hjw/Datasets/SemKITTI-DVPS/video_sequence/train/"
    # gt_dir = "/hjw/Datasets/SemKITTI-DVPS/video_sequence/train/"

    # depth_thres = [-1, 0.5, 0.25, 0.1]
    # depth_thres = [0.5]
    depth_thres = [-1]
    for d in depth_thres:
        main(1, pred_dir, gt_dir, d)
