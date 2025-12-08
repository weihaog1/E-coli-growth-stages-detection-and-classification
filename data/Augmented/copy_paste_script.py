import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import torch
import numpy as np
from xml.dom import minidom

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget

CLASS_TO_ID = {
    "Rod": 0,
    "Dividing": 1,
    "Microcolony": 2
}

ID_TO_CLASS = {
    0: "Rod",
    1: "Dividing",
    2: "Microcolony"
}

# Parse XML files

def xml_to_df(path):
    """
    Convert all XML annotation files in a directory into a single pandas dataframe.
    """
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Each object has a bounding box
        for member in root.findall('object'):
            value = (root.find('filename').text,               # image filename
                    int(root.find('size')[0].text),           # image width
                    int(root.find('size')[1].text),           # image height
                    member[0].text,                           # object class name
                    int(member[4][0].text),                   # xmin
                    int(member[4][1].text),                   # ymin
                    int(member[4][2].text),                   # xmax
                    int(member[4][3].text)                    # ymax
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def load_annotations(image_dir):
    """
    Load both the training and testing annotation directories.
    Returns: train_df, test_df
    """
    dfs = [] 
    for folder in ['train_annotations', 'test_annotations']:
        ann_path = os.path.join(image_dir, folder)
        xml_df = xml_to_df(ann_path)
        dfs.append(xml_df)
    return dfs

def write_pascal_xml(filename, folder, path, width, height, depth, boxes, labels, label_map, save_path):
    """
    Write a single Pascal VOC XML annotation file.
    """
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = "0"
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        name = label_map[int(label.item())]
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)
    
    xml_str = ET.tostring(annotation, encoding="unicode")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="\t")
    # Remove XML declaration line
    pretty_xml_lines = pretty_xml.split('\n')
    if pretty_xml_lines[0].startswith('<?xml'):
        pretty_xml = '\n'.join(pretty_xml_lines[1:])
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

# Data Processing

def boxes_to_masks(boxes, height, width):
    """
    Create masks from bounding boxes.
    Each box becomes a filled rectangle mask.
    Returns: tensor of shape (N, H, W)
    """
    masks = []
    for (xmin, ymin, xmax, ymax) in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = 1
        masks.append(mask)
    masks_t = torch.tensor(np.stack(masks), dtype=torch.uint8)
    return masks_t

def df_to_detection_targets(df, image_root):
    """
    Convert DataFrame annotations to DetectionTarget objects for segpaste
    """
    targets = []
    grouped = df.groupby("filename")

    for filename, group in grouped:
        img_path = os.path.join(image_root, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            continue
        # Convert to RGB for segpaste
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        # Normalize and convert to CHW format
        img_t = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
        # Extract boxes
        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        # Extract labels
        labels = group["class"].map(CLASS_TO_ID).values
        labels_t = torch.tensor(labels, dtype=torch.int64)
        # Create masks from boxes
        masks_t = boxes_to_masks(boxes, H, W)
        
        target = DetectionTarget(
            image=img_t,
            boxes=boxes_t,
            labels=labels_t,
            masks=masks_t
        )
        targets.append(target)

    return targets

def augment_single_object(source):
    """
    Apply augmentation to a single object crop.
    Expects source DetectionTarget to contain exactly one object.
    """
    box = source.boxes[0]
    label = source.labels[0]
    mask = source.masks[0]
    x1, y1, x2, y2 = box.int().tolist()

    # Crop
    obj_img = source.image[:, y1:y2, x1:x2]
    obj_mask = mask[y1:y2, x1:x2]
    obj_img_np = (obj_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    obj_mask_np = obj_mask.numpy().astype(np.uint8)
    
    # Augmentations 
    if np.random.rand() < 0.8:
        k = np.random.randint(1, 4)
        obj_img_np = np.rot90(obj_img_np, k)
        obj_mask_np = np.rot90(obj_mask_np, k)

    if np.random.rand() < 0.8:
        obj_img_np = np.fliplr(obj_img_np)
        obj_mask_np = np.fliplr(obj_mask_np)

    scale = np.random.uniform(0.7, 1.3)
    new_h = int(obj_img_np.shape[0] * scale)
    new_w = int(obj_img_np.shape[1] * scale)
    obj_img_np = cv2.resize(obj_img_np, (new_w, new_h))
    obj_mask_np = cv2.resize(obj_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    max_area = 8100  # pixel area limit

    current_area = new_w * new_h
    if current_area > max_area:
        shrink_factor = (max_area / current_area) ** 0.5  # sqrt keeps aspect ratio
        new_w = max(1, int(new_w * shrink_factor))
        new_h = max(1, int(new_h * shrink_factor))
        obj_img_np = cv2.resize(obj_img_np, (new_w, new_h))
        obj_mask_np = cv2.resize(obj_mask_np, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
    # Back to torch
    new_img = torch.tensor(obj_img_np / 255.0, dtype=torch.float32).permute(2, 0, 1)
    new_mask = torch.tensor(obj_mask_np, dtype=torch.uint8)
    new_box = torch.tensor([0, 0, new_w, new_h], dtype=torch.float32)

    return DetectionTarget(
        image=new_img,
        boxes=new_box.unsqueeze(0),
        labels=label.unsqueeze(0),
        masks=new_mask.unsqueeze(0)
    )


def get_class_counts(df):
    """
    Get the count of each class in the dataset.
    """
    return df['class'].value_counts().to_dict()

# Copy-Paste Augmentation

def run_copy_paste(train_df, image_root, output_dir, target_count=1000):
    """
    Copy paste augmentation to balance classes to target_count objects each.
    Args:
        train_df: DataFrame with training annotations
        image_root: Directory containing training images
        output_dir: Directory to save augmented images and annotations
        target_count: Target number of objects per class (default: 1000)
    """
    # Setup output directories
    img_out = os.path.join(output_dir, "train_images")
    ann_out = os.path.join(output_dir, "train_annotations")
    
    if os.path.exists(ann_out):
        print("Augmented data already exists, skipping copy-paste augmentation")
        return
    
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)
    
    # Build label mappings
    class_names = list(CLASS_TO_ID.keys())
    id_to_name = ID_TO_CLASS
    
    # Current class distribution
    class_counts = get_class_counts(train_df)
    print(f"Raw class distribution: {class_counts}")
    classes_needed = {cls: max(0, target_count - count) for cls, count in class_counts.items()}
    print(f"Objects needed per class: {classes_needed}")
    
    # Load dataset
    targets = df_to_detection_targets(train_df, image_root)
    print(f"Loaded {len(targets)} training samples")
    
    # Group targets by dominant class
    targets_by_class = {cls: [] for cls in class_names}
    for idx, target in enumerate(targets):
        label_counts = {}
        for label in target.labels:
            cls_name = id_to_name[int(label.item())]
            label_counts[cls_name] = label_counts.get(cls_name, 0) + 1
        if label_counts:
            dominant_class = max(label_counts, key=label_counts.get)
            targets_by_class[dominant_class].append((idx, target))
    
    # Configure copy paste augmentation
    config = CopyPasteConfig(
        paste_probability=1.0,  # Every image pasted
        max_paste_objects=4,    # 2-4 sources
        min_paste_objects=2,
        scale_range=(0.3, 1.1),
        occluded_area_threshold = 0.1
    )
    cp = CopyPasteAugmentation(config)
        
    # Track counts and iterate until balanced
    current_counts = class_counts.copy()
    aug_counter = 0
    
    while any(current_counts.get(cls, 0) < target_count for cls in class_names):
        # Find class needing most augmentation
        most_needed_class = min(class_names, key=lambda x: current_counts.get(x, 0))
        if current_counts.get(most_needed_class, 0) >= target_count:
            break
        
        # Select target and source images containing needed class
        if targets_by_class[most_needed_class]:
            idx, t = targets_by_class[most_needed_class][aug_counter % len(targets_by_class[most_needed_class])]
            
            source_candidates = targets_by_class[most_needed_class]
            sources = []
            for i in range(min(4, len(source_candidates))):
                src_idx = (idx + i + 1) % len(source_candidates)
                original = source_candidates[src_idx][1]
                num_objects = min(2, len(original.labels))
                perm = torch.randperm(len(original.labels))[:num_objects]
                limited_source = DetectionTarget(
                    image=original.image,
                    boxes=original.boxes[perm],
                    labels=original.labels[perm],
                    masks=original.masks[perm]
                )
                # Limit to 1 object copied per source
                for j in range(len(limited_source.boxes)):
                    single = DetectionTarget(
                        image=original.image,
                        boxes=limited_source.boxes[j:j+1],
                        labels=limited_source.labels[j:j+1],
                        masks=limited_source.masks[j:j+1]
                    )
                    single = augment_single_object(single)
                    sources.append(single)
        else:
            # Random selection otherwise
            idx = aug_counter % len(targets)
            t = targets[idx]
            sources = targets[idx + 1: idx + 3] if idx + 3 < len(targets) else targets[:2]
        
        # Apply augmentation
        try:
            aug = cp.transform(t, sources)
            for label in aug.labels:
                cls_name = id_to_name[int(label.item())]
                current_counts[cls_name] = current_counts.get(cls_name, 0) + 1
        
        except Exception as e:
            print(f"Failed on sample {aug_counter}: {e}")
            aug_counter += 1
            continue
        
        # Save augmented image
        out_img_name = f"aug_{aug_counter}.png"
        out_img_path = os.path.join(img_out, out_img_name)
        aug_img = (aug.image * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        cv2.imwrite(out_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)) # Output grayscale
        
        # Save XML annotation
        H, W = aug_img.shape[:2]
        out_xml_path = os.path.join(ann_out, f"aug_{aug_counter}.xml")
        write_pascal_xml(
            filename=out_img_name,
            folder="train_images",
            path=out_img_path,
            width=W,
            height=H,
            depth=1,
            boxes=aug.boxes,
            labels=aug.labels,
            label_map=id_to_name,
            save_path=out_xml_path
        )
        aug_counter += 1
        
        # Safety check to prevent infinite loop
        if aug_counter > 1000:
            print("Reached maximum augmentation limit")
            break
    
    print(f"Final class distribution: {current_counts}")
    print(f"Generated {aug_counter} augmented images + XML saved to {output_dir}")

def main():
    # Configuration
    CLASSES = ["Rod", "Dividing", "Microcolony"]
    TARGET_COUNT = 800
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "../raw/image_patches")
    train_image_dir = os.path.join(image_dir, "train_images")
    augmented_dir = os.path.join(BASE_DIR, "augmented_images")
    
    # Load annotations
    train_df, test_df = load_annotations(image_dir)
    
    # Run copy-paste augmentation
    run_copy_paste(train_df, train_image_dir, augmented_dir, target_count=TARGET_COUNT)

main()
