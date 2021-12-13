from typing import List, Tuple

import cv2
import numpy as np
import torch


def compute_iou(labels: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:]


def precision_at(threshold: float, iou: np.ndarray) -> Tuple[int, int, int]:
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1
    false_negatives = np.sum(matches, axis=1) == 0
    false_positives = np.sum(matches, axis=0) == 0
    return true_positives.sum(), false_positives.sum(), false_negatives.sum()


def iou_map(truths: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    preds = [cv2.connectedComponents((x > 0.5).numpy().astype(np.uint8))[1] for x in preds]
    truths = [cv2.connectedComponents(x.numpy().astype(np.uint8))[1] for x in truths]
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]
    prec: List[float] = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn
        p = tps / (tps + fps + fns)
        prec.append(p)
    return torch.tensor(np.mean(prec))
