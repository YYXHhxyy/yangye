"""
Tianzhibei Competition - YOLO11x-OBB Inference Script
Usage (in Docker container):
    python run.py /input_path /output_path

Input:  /input_path  - Directory containing test TIF images
Output: /output_path - Directory containing result XML files (one per image)
"""
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from ultralytics import YOLO
import tifffile

# Class name mapping (must match training)
CLASS_NAMES = [
    "Small Car",       # 0
    "Bus",             # 1
    "Cargo Truck",     # 2
    "Dump Truck",      # 3
    "Van",             # 4
    "Trailer",         # 5
    "Tractor",         # 6
    "Excavator",       # 7
    "Truck Tractor",   # 8
    "other-vehicle",   # 9
]

CONF_THRESHOLD = 0.25


def read_tif_info(tif_path):
    """Read TIF image, return (rgb_img, original_depth)."""
    img = tifffile.imread(tif_path)
    if img.ndim == 2:
        depth = 1
        rgb = np.stack([img] * 3, axis=-1)
    else:
        depth = img.shape[2]
        if depth >= 4:
            rgb = img[:, :, :3]
        elif depth == 3:
            rgb = img
        else:
            rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
            depth = 1
    return rgb, depth


def build_xml_annotation(filename, width, height, depth, detections):
    """Build XML string matching the competition format."""
    root = ET.Element("annotation")

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    objects_elem = ET.SubElement(root, "objects")
    for det in detections:
        obj = ET.SubElement(objects_elem, "object")
        ET.SubElement(obj, "coordinate").text = "pixel"
        ET.SubElement(obj, "type").text = "rectangle"
        ET.SubElement(obj, "description").text = "None"

        possibleresult = ET.SubElement(obj, "possibleresult")
        ET.SubElement(possibleresult, "name").text = det["class_name"]

        points = ET.SubElement(obj, "points")
        for x, y in det["points"]:
            ET.SubElement(points, "point").text = f"{x:.6f},{y:.6f}"

    rough_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough_str)
    pretty = dom.toprettyxml(indent="\t")
    pretty = pretty.replace(
        '<?xml version="1.0" ?>',
        "<?xml version='1.0' encoding='utf-8'?>"
    )
    return pretty


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))

    tif_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith((".tif", ".tiff"))]
    tif_files.sort()

    for tif_name in tif_files:
        tif_path = os.path.join(input_dir, tif_name)
        img, original_depth = read_tif_info(tif_path)
        height, width = img.shape[:2]

        results = model(img, conf=CONF_THRESHOLD)[0]

        detections = []
        if results.obb is not None:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            cls_ids = results.obb.cls.cpu().numpy().astype(int)

            for i in range(len(xyxyxyxy)):
                pixel_points = np.round(xyxyxyxy[i]).astype(int)
                points_list = list(pixel_points) + [pixel_points[0]]

                detections.append({
                    "class_name": CLASS_NAMES[cls_ids[i]],
                    "points": [(float(x), float(y)) for x, y in points_list],
                })

        xml_name = os.path.splitext(tif_name)[0] + ".xml"
        xml_str = build_xml_annotation(
            filename=tif_name, width=width, height=height,
            depth=original_depth, detections=detections,
        )

        with open(os.path.join(output_dir, xml_name), "w", encoding="utf-8") as f:
            f.write(xml_str)

        print(f"  {tif_name}: {len(detections)} objects detected (depth={original_depth})")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
