import os
import cv2
import glob
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR = os.path.join(BASE_DIR, "augmented_images")
SOURCE_IMAGE_DIR = os.path.join(SOURCE_DIR, "train_images")
SOURCE_ANNOTATION_DIR = os.path.join(SOURCE_DIR, "train_annotations")

OUTPUT_DIR = os.path.join(BASE_DIR, "full_augmented_dataset")
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "train_images")
OUTPUT_ANNOTATION_DIR = os.path.join(OUTPUT_DIR, "train_annotations")

def read_pascal_xml(xml_path):
    """
    Read a Pascal VOC XML file and extract image info and boxes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    bounding_boxes = []
    labels = []

    for obj in root.findall("object"):
        label_name = obj.find("name").text

        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        bounding_boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_name)

    return filename, image_width, image_height, bounding_boxes, labels

def write_pascal_xml(
    filename,
    folder,
    path,
    width,
    height,
    depth,
    boxes,
    labels,
    label_map,
    save_path
):
    """
    Write a single Pascal VOC XML annotation file.
    This version matches the exact format used in the copy-paste pipeline,
    but accepts plain Python boxes + string labels.
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
        x1, y1, x2, y2 = map(int, box)
        if isinstance(label, str):
            name = label
        else:
            name = label_map[int(label)]

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

    pretty_xml_lines = pretty_xml.split('\n')
    if pretty_xml_lines[0].startswith('<?xml'):
        pretty_xml = '\n'.join(pretty_xml_lines[1:])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

def rotate_90_boxes(bounding_boxes, image_width, image_height):
    return [[
        image_height - ymax,
        xmin,
        image_height - ymin,
        xmax
    ] for xmin, ymin, xmax, ymax in bounding_boxes]


def rotate_180_boxes(bounding_boxes, image_width, image_height):
    return [[image_width - xmax, image_height - ymax,
             image_width - xmin, image_height - ymin]
            for xmin, ymin, xmax, ymax in bounding_boxes]


def rotate_270_boxes(bounding_boxes, image_width, image_height):
    return [[
        ymin,
        image_width - xmax,
        ymax,
        image_width - xmin
    ] for xmin, ymin, xmax, ymax in bounding_boxes]


def flip_horizontal_boxes(bounding_boxes, image_width):
    return [[image_width - xmax, ymin,
             image_width - xmin, ymax]
            for xmin, ymin, xmax, ymax in bounding_boxes]


def flip_vertical_boxes(bounding_boxes, image_height):
    return [[xmin, image_height - ymax,
             xmax, image_height - ymin]
            for xmin, ymin, xmax, ymax in bounding_boxes]

def check_output_directory():
    """
    Delete and recreate the full augmented output directory.
    """
    if os.path.exists(OUTPUT_DIR):
        print("Output Directory already exists, skipping augmentations")
    else:
        os.makedirs(OUTPUT_IMAGE_DIR)
        os.makedirs(OUTPUT_ANNOTATION_DIR)

def run_geometric_full_augmentation():
    """
    Apply fixed geometric augmentations to every training image and XML.
    """
    check_output_directory()

    xml_files = glob.glob(os.path.join(SOURCE_ANNOTATION_DIR, "*.xml"))
    print(f"Found {len(xml_files)} source images")

    generated_count = 0

    for xml_path in xml_files:
        filename, image_width, image_height, bounding_boxes, labels = read_pascal_xml(xml_path)

        image_path = os.path.join(SOURCE_IMAGE_DIR, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skipping missing image: {image_path}")
            continue

        base_name = os.path.splitext(filename)[0]

        augmentation_map = {
            "rot90": (
                cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                rotate_90_boxes(bounding_boxes, image_width, image_height),
                image_height, image_width
            ),
            "rot180": (
                cv2.rotate(image, cv2.ROTATE_180),
                rotate_180_boxes(bounding_boxes, image_width, image_height),
                image_width, image_height
            ),
            "rot270": (
                cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
                rotate_270_boxes(bounding_boxes, image_width, image_height),
                image_height, image_width
            ),
            "flip_horizontal": (
                cv2.flip(image, 1),
                flip_horizontal_boxes(bounding_boxes, image_width),
                image_width, image_height
            ),
            "flip_vertical": (
                cv2.flip(image, 0),
                flip_vertical_boxes(bounding_boxes, image_height),
                image_width, image_height
            ),
        }

        # Save original image + XML first
        original_image_output_path = os.path.join(OUTPUT_IMAGE_DIR, filename)
        original_xml_output_path = os.path.join(OUTPUT_ANNOTATION_DIR, filename.replace(".png", ".xml"))

        cv2.imwrite(original_image_output_path, image)

        write_pascal_xml(
            filename=filename,
            folder="train_images",
            path=original_image_output_path,
            width=image_width,
            height=image_height,
            depth=1,
            boxes=bounding_boxes,
            labels=labels,
            label_map=None,    
            save_path=original_xml_output_path
        )


        generated_count += 1

        # Save all geometric variants
        for augmentation_tag, (augmented_image,
                                 augmented_boxes,
                                 new_width,
                                 new_height) in augmentation_map.items():

            output_image_name = f"{base_name}_{augmentation_tag}.png"
            output_xml_name = f"{base_name}_{augmentation_tag}.xml"

            output_image_path = os.path.join(OUTPUT_IMAGE_DIR, output_image_name)
            output_xml_path = os.path.join(OUTPUT_ANNOTATION_DIR, output_xml_name)

            cv2.imwrite(output_image_path, augmented_image)

            write_pascal_xml(
                filename=output_image_name,          
                folder="train_images",
                path=output_image_path,             
                width=new_width,                  
                height=new_height,             
                depth=1,
                boxes=augmented_boxes,             
                labels=labels,
                label_map=None,      
                save_path=output_xml_path      
            )


            generated_count += 1

    print(f"Full augmented dataset created: {generated_count} images + XML files")

run_geometric_full_augmentation()