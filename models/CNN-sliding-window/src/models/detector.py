"""
Sliding window detector using patch CNN.

For inference: slides a window across the image, classifies each patch,
then applies non-maximum suppression to get final detections.
"""

import torch
import numpy as np
from typing import List, Tuple


class SlidingWindowDetector:
    """
    Object detector using sliding window + CNN classifier.

    Workflow:
    1. Slide window across image
    2. Classify each patch
    3. Keep patches with non-background predictions
    4. Apply NMS to merge overlapping detections
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int = 64,
        stride: int = 16,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.patch_size = patch_size
        self.stride = stride
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run detection on a single grayscale image.

        Args:
            image: numpy array of shape (H, W), values 0-255

        Returns:
            List of detections: {"bbox": (x1,y1,x2,y2), "class_id": int, "confidence": float}
        """
        h, w = image.shape
        detections = []

        # Collect all patches
        patches = []
        positions = []

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y : y + self.patch_size, x : x + self.patch_size]
                patches.append(patch)
                positions.append((x, y))

        if not patches:
            return []

        # Batch inference
        patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32)
        patches_tensor = patches_tensor.unsqueeze(1) / 255.0  # (N, 1, H, W)
        patches_tensor = (patches_tensor - 0.5) / 0.5  # Normalize
        patches_tensor = patches_tensor.to(self.device)

        logits = self.model(patches_tensor)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)

        # Filter: keep non-background with high confidence
        for i, (x, y) in enumerate(positions):
            class_id = predictions[i].item()
            conf = confidences[i].item()

            if class_id != 3 and conf > self.conf_threshold:
                detections.append({
                    "bbox": (x, y, x + self.patch_size, y + self.patch_size),
                    "class_id": class_id,
                    "confidence": conf,
                })

        # Apply NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[dict]) -> List[dict]:
        """
        Non-maximum suppression to remove overlapping boxes.
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove boxes with high IoU
            remaining = []
            for det in detections:
                if self._iou(best["bbox"], det["bbox"]) < self.nms_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def _iou(self, box_a: Tuple, box_b: Tuple) -> float:
        """Compute IoU between two boxes."""
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
