"""
Evaluation metrics for semantic segmentation.

Includes IoU (Jaccard), Dice, pixel accuracy, and bbox extraction.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy import ndimage


def compute_iou_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4
) -> Dict[int, float]:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to IoU score
    """
    iou_per_class = {}

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            # Class not present in either pred or target
            iou_per_class[cls] = float('nan')
        else:
            iou_per_class[cls] = intersection / union

    return iou_per_class


def compute_mean_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
    ignore_background: bool = False
) -> float:
    """
    Compute mean IoU across all classes.

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes
        ignore_background: If True, exclude class 0 from mean

    Returns:
        Mean IoU score
    """
    iou_per_class = compute_iou_per_class(pred, target, num_classes)

    start_class = 1 if ignore_background else 0
    valid_ious = [iou for cls, iou in iou_per_class.items()
                  if cls >= start_class and not np.isnan(iou)]

    if len(valid_ious) == 0:
        return 0.0

    return np.mean(valid_ious)


def compute_dice_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4
) -> Dict[int, float]:
    """
    Compute Dice coefficient for each class.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to Dice score
    """
    dice_per_class = {}

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()

        if total == 0:
            dice_per_class[cls] = float('nan')
        else:
            dice_per_class[cls] = 2 * intersection / total

    return dice_per_class


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute overall pixel accuracy.

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)

    Returns:
        Pixel accuracy (correct pixels / total pixels)
    """
    correct = (pred == target).sum()
    total = pred.size
    return correct / total


def compute_class_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4
) -> Dict[int, float]:
    """
    Compute pixel accuracy for each class.

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to accuracy
    """
    acc_per_class = {}

    for cls in range(num_classes):
        target_mask = (target == cls)
        if target_mask.sum() == 0:
            acc_per_class[cls] = float('nan')
        else:
            correct = np.logical_and(pred == cls, target_mask).sum()
            acc_per_class[cls] = correct / target_mask.sum()

    return acc_per_class


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        pred: Predicted segmentation mask (H, W)
        target: Ground truth mask (H, W)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes, num_classes)
        where cm[i, j] = number of pixels with true class i predicted as class j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_cls in range(num_classes):
        for pred_cls in range(num_classes):
            cm[true_cls, pred_cls] = np.logical_and(
                target == true_cls,
                pred == pred_cls
            ).sum()

    return cm


def masks_to_bboxes(mask: np.ndarray, min_area: int = 10) -> List[Dict]:
    """
    Extract bounding boxes from segmentation mask using connected components.

    Args:
        mask: Segmentation mask (H, W) with class IDs
        min_area: Minimum area for a valid detection

    Returns:
        List of detections: [{"bbox": (x1,y1,x2,y2), "class_id": int, "area": int}]
    """
    detections = []

    # Process each non-background class
    for class_id in range(1, 4):  # 1=rod, 2=dividing, 3=microcolony
        class_mask = (mask == class_id).astype(np.uint8)

        if class_mask.sum() == 0:
            continue

        # Find connected components
        labeled, num_features = ndimage.label(class_mask)

        for i in range(1, num_features + 1):
            component = (labeled == i)
            area = component.sum()

            if area < min_area:
                continue

            # Get bounding box
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            detections.append({
                "bbox": (int(x1), int(y1), int(x2 + 1), int(y2 + 1)),
                "class_id": class_id,
                "area": int(area)
            })

    return detections


def compute_detection_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    iou_threshold: float = 0.5,
    min_area: int = 10
) -> Dict:
    """
    Compute detection metrics (precision, recall, F1) from segmentation masks.

    Extracts bounding boxes from masks and computes matching.

    Args:
        pred_mask: Predicted segmentation mask
        target_mask: Ground truth segmentation mask
        iou_threshold: IoU threshold for matching
        min_area: Minimum area for valid detection

    Returns:
        Dictionary with precision, recall, F1 per class
    """
    pred_boxes = masks_to_bboxes(pred_mask, min_area)
    target_boxes = masks_to_bboxes(target_mask, min_area)

    metrics = {}

    for class_id in range(1, 4):
        pred_cls = [b for b in pred_boxes if b["class_id"] == class_id]
        target_cls = [b for b in target_boxes if b["class_id"] == class_id]

        if len(target_cls) == 0 and len(pred_cls) == 0:
            metrics[class_id] = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            continue

        if len(target_cls) == 0:
            metrics[class_id] = {"precision": 0.0, "recall": 1.0, "f1": 0.0}
            continue

        if len(pred_cls) == 0:
            metrics[class_id] = {"precision": 1.0, "recall": 0.0, "f1": 0.0}
            continue

        # Match predictions to targets
        matched_targets = set()
        tp = 0

        for pred in pred_cls:
            best_iou = 0
            best_target_idx = -1

            for idx, target in enumerate(target_cls):
                if idx in matched_targets:
                    continue

                iou = _box_iou(pred["bbox"], target["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = idx

            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp += 1
                matched_targets.add(best_target_idx)

        precision = tp / len(pred_cls) if len(pred_cls) > 0 else 0
        recall = tp / len(target_cls) if len(target_cls) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[class_id] = {"precision": precision, "recall": recall, "f1": f1}

    return metrics


def _box_iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


class SegmentationMetrics:
    """
    Accumulator for computing metrics over a dataset.

    Accumulates predictions and computes aggregate metrics.
    """

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["background", "rod", "dividing", "microcolony"]
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics with a new prediction.

        Args:
            pred: Predicted mask (H, W) or (B, H, W)
            target: Ground truth mask (H, W) or (B, H, W)
        """
        if pred.ndim == 3:
            # Batch of predictions
            for p, t in zip(pred, target):
                self._update_single(p, t)
        else:
            self._update_single(pred, target)

    def _update_single(self, pred: np.ndarray, target: np.ndarray):
        """Update with single prediction."""
        cm = compute_confusion_matrix(pred, target, self.num_classes)
        self.confusion_matrix += cm
        self.total_pixels += pred.size

    def compute(self) -> Dict:
        """
        Compute all metrics from accumulated confusion matrix.

        Returns:
            Dictionary with all metrics
        """
        # Per-class IoU from confusion matrix
        iou_per_class = {}
        dice_per_class = {}

        for cls in range(self.num_classes):
            tp = self.confusion_matrix[cls, cls]
            fp = self.confusion_matrix[:, cls].sum() - tp
            fn = self.confusion_matrix[cls, :].sum() - tp

            # IoU
            union = tp + fp + fn
            iou_per_class[cls] = tp / union if union > 0 else float('nan')

            # Dice
            dice_denom = 2 * tp + fp + fn
            dice_per_class[cls] = 2 * tp / dice_denom if dice_denom > 0 else float('nan')

        # Mean IoU (excluding classes not present)
        valid_ious = [v for v in iou_per_class.values() if not np.isnan(v)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0

        # Mean IoU excluding background
        valid_ious_no_bg = [v for k, v in iou_per_class.items() if k > 0 and not np.isnan(v)]
        mean_iou_no_bg = np.mean(valid_ious_no_bg) if valid_ious_no_bg else 0.0

        # Pixel accuracy
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return {
            "pixel_accuracy": pixel_acc,
            "mean_iou": mean_iou,
            "mean_iou_no_bg": mean_iou_no_bg,
            "iou_per_class": {self.class_names[k]: v for k, v in iou_per_class.items()},
            "dice_per_class": {self.class_names[k]: v for k, v in dice_per_class.items()},
            "confusion_matrix": self.confusion_matrix
        }
