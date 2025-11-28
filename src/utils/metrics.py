"""
Evaluation metrics for classification and detection.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> dict:
    """
    Compute classification metrics.
    
    Returns:
        {
            "precision": float (macro),
            "recall": float (macro),
            "f1": float (macro),
            "confusion_matrix": np.ndarray,
            "per_class": dict with per-class metrics
        }
    """
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(num_classes)),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute average precision for a single class.
    
    Args:
        predictions: [{"bbox": (x1,y1,x2,y2), "confidence": float}]
        ground_truths: [{"bbox": (x1,y1,x2,y2)}]
        iou_threshold: IoU threshold for a match
    
    Returns:
        Average precision value
    """
    if not ground_truths:
        return 0.0 if predictions else 1.0
    
    if not predictions:
        return 0.0
    
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    
    matched = [False] * len(ground_truths)
    tp = []
    fp = []
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if matched[gt_idx]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision and recall at each threshold
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(ground_truths)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for r_threshold in np.linspace(0, 1, 11):
        precisions_above = precision[recall >= r_threshold]
        if len(precisions_above) > 0:
            ap += precisions_above.max() / 11
    
    return ap


def compute_map(
    all_predictions: Dict[int, List[Dict]],
    all_ground_truths: Dict[int, List[Dict]],
    iou_threshold: float = 0.5,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mean average precision across all classes.
    
    Args:
        all_predictions: {class_id: [predictions]}
        all_ground_truths: {class_id: [ground_truths]}
    
    Returns:
        (mAP, {class_id: AP})
    """
    aps = {}
    
    all_classes = set(all_predictions.keys()) | set(all_ground_truths.keys())
    
    for class_id in all_classes:
        preds = all_predictions.get(class_id, [])
        gts = all_ground_truths.get(class_id, [])
        aps[class_id] = compute_ap(preds, gts, iou_threshold)
    
    mAP = np.mean(list(aps.values())) if aps else 0.0
    
    return mAP, aps

