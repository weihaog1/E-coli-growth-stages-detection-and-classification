"""
Annotation format converters and utilities.

Parses Pascal VOC XML annotations for the DeepBacs E. coli dataset.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_TO_ID = {
    "rod": 0,
    "dividing": 1,
    "microcolony": 2,
}


def parse_xml(xml_path: str) -> dict:
    """
    Parse a Pascal VOC XML file.

    Returns:
        {
            "width": int,
            "height": int,
            "objects": [{"class_name": str, "class_id": int, "bbox": (x1,y1,x2,y2)}]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        box = obj.find("bndbox")

        bbox = (
            float(box.find("xmin").text),
            float(box.find("ymin").text),
            float(box.find("xmax").text),
            float(box.find("ymax").text),
        )

        objects.append({
            "class_name": name,
            "class_id": CLASS_TO_ID.get(name.lower(), -1),
            "bbox": bbox,
        })

    return {"width": width, "height": height, "objects": objects}


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Compute intersection over union between two boxes.
    Boxes are (x1, y1, x2, y2) format.
    """
    # Intersection coordinates
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    # No intersection
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union
