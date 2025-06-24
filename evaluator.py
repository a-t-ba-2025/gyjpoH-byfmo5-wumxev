import csv
import json
import os
from collections import defaultdict
from shapely.geometry import box as shapely_box
from tqdm import tqdm
from dotenv import load_dotenv

from Evaluator_DocLayNet.summary import summarize_doclaynet

# Load environment variables from .env file
load_dotenv()

# Read paths from environment
GT_JSON = os.getenv("GT_JSON")  # Path to ground truth annotations
PRED_FOLDER = os.getenv("PRED_FOLDER")  # Path to folder with model predictions

if not GT_JSON or not PRED_FOLDER:
    raise ValueError("Environment variables GT_JSON or PRED_FOLDER are not set!")


def compute_iou(box1, box2):
    inter = box1.intersection(box2).area
    union = box1.union(box2).area
    return inter / union if union > 0 else 0


def evaluate_doclaynet(iou_threshold):
    categories = {
        1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
        5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
        9: "Table", 10: "Text", 11: "Title"
    }

    for category in categories.values():
        print(f"\nEvaluating category: {category}")
        evaluate_coco(GT_JSON, PRED_FOLDER, iou_threshold, category_filter=category)

    print("\nEvaluating ALL categories")
    evaluate_coco(GT_JSON, PRED_FOLDER, iou_threshold, category_filter=None)
    summarize_doclaynet(iou_threshold)


def evaluate_coco(coco_json_path, pred_folder, iou_threshold, category_filter=None):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    gt_boxes_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        cat_name = category_id_to_name[ann["category_id"]]
        if category_filter and cat_name != category_filter:
            continue
        file_name = image_id_to_file[ann["image_id"]]
        x1, y1, w, h = ann["bbox"]
        gt_boxes_per_image[file_name].append(shapely_box(x1, y1, x1 + w, y1 + h))

    sources = ["DETR", "FRCNN", "ALL"]
    results_by_source = {s: [] for s in sources}

    for filename in tqdm(os.listdir(pred_folder)):
        if not filename.endswith(".json"):
            continue

        file_name = filename.replace(".json", ".png")
        with open(os.path.join(pred_folder, filename), "r", encoding="utf-8") as f:
            preds = json.load(f)

        gt_boxes = gt_boxes_per_image.get(file_name, [])

        for source in sources:
            pred_boxes = [shapely_box(*p["box"]) for p in preds
                          if (source == "ALL" or p.get("source") == source)
                          and (not category_filter or p.get("label_name") == category_filter)]

            tp, fp, matched = 0, 0, set()
            for pred_box in pred_boxes:
                match_found = any(i not in matched and compute_iou(pred_box, gt_box) >= iou_threshold
                                  for i, gt_box in enumerate(gt_boxes))
                if match_found:
                    matched.add(next(i for i, gt_box in enumerate(gt_boxes)
                                     if i not in matched and compute_iou(pred_box, gt_box) >= iou_threshold))
                    tp += 1
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            results_by_source[source].append({
                "image": file_name, "TP": tp, "FP": fp, "FN": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3)
            })

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    eval_dir = os.path.join(project_root, "Evaluator_DocLayNet", f"IOU_{int(iou_threshold * 100):02d}")
    os.makedirs(eval_dir, exist_ok=True)

    for source in sources:
        results = results_by_source[source]
        csv_file = os.path.join(eval_dir, f"DOCLAYNET_evaluation_{source}_{category_filter or 'ALL'}.csv")

        with open(csv_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"])
            for r in results:
                writer.writerow([r[k] for k in ["image", "TP", "FP", "FN", "precision", "recall", "f1_score"]])

            if results:
                avg = {key: round(sum(r[key] for r in results) / len(results), 3)
                       for key in ["precision", "recall", "f1_score"]}
                writer.writerow(["AVERAGE", "", "", "", avg["precision"], avg["recall"], avg["f1_score"]])

        print(f"Results saved to: {csv_file}")