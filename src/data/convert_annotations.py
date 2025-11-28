"""
Annotation format converters.

Converts Pascal VOC XML annotations to:
- YOLO format (for YOLOv8)
- Patch labels (for sliding window CNN)
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
            "class_id": CLASS_TO_ID.get(name, -1),
            "bbox": bbox,
        })
    
    return {"width": width, "height": height, "objects": objects}


def xml_to_yolo(xml_path: str, output_path: str) -> None:
    """
    Convert XML to YOLO format.
    
    YOLO format per line: class_id x_center y_center width height
    All coordinates normalized to [0, 1].
    """
    data = parse_xml(xml_path)
    img_w, img_h = data["width"], data["height"]
    
    lines = []
    for obj in data["objects"]:
        if obj["class_id"] == -1:
            continue
        
        x1, y1, x2, y2 = obj["bbox"]
        
        # Convert to center + size, normalized
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        
        # Clamp to valid range
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        lines.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


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


def xml_to_patches(xml_path: str, patch_size: int = 64, stride: int = 32) -> list:
    """
    Generate patch labels for sliding window approach.
    
    Returns:
        List of {"bbox": (x1,y1,x2,y2), "class_id": int}
        class_id = 3 means background
    """
    data = parse_xml(xml_path)
    img_w, img_h = data["width"], data["height"]
    
    patches = []
    
    for y in range(0, img_h - patch_size + 1, stride):
        for x in range(0, img_w - patch_size + 1, stride):
            patch_box = (x, y, x + patch_size, y + patch_size)
            
            # Find best matching object
            best_iou = 0.0
            best_class = 3  # Background
            
            for obj in data["objects"]:
                iou = compute_iou(patch_box, obj["bbox"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_class = obj["class_id"]
            
            patches.append({"bbox": patch_box, "class_id": best_class})
    
    return patches


def convert_dataset(input_dir: str, output_dir: str) -> None:
    """
    Batch convert all XML files in a directory to YOLO format.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(input_path.glob("*.xml"))
    
    for xml_file in xml_files:
        out_file = output_path / f"{xml_file.stem}.txt"
        xml_to_yolo(str(xml_file), str(out_file))
    
    print(f"Converted {len(xml_files)} files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO format")
    parser.add_argument("--input", required=True, help="Directory with XML files")
    parser.add_argument("--output", required=True, help="Output directory for YOLO labels")
    args = parser.parse_args()
    
    convert_dataset(args.input, args.output)
