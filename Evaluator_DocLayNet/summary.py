import os
import pandas as pd
from collections import defaultdict

def summarize_doclaynet(iou_threshold: int):
    print(f"\n--- Calculate metrics from CSVs for IOU={iou_threshold} ---\n")

    base_output_folder = os.path.join(os.path.dirname(__file__), f"IOU_{iou_threshold}")
    os.makedirs(base_output_folder, exist_ok=True)

    all_files = [f for f in os.listdir(base_output_folder) if f.endswith("_ALL.csv")]
    if not all_files:
        print(f"No *_ALL.csv found in IOU_{iou_threshold} – skip.")
        return

    model_scores = defaultdict(lambda: defaultdict(list))
    label_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in all_files:
        filepath = os.path.join(base_output_folder, filename)
        df = pd.read_csv(filepath)
        if "Image" not in df.columns:
            print(f"Column “Image” is missing in {filename} – no mAP/AP possible.")
            continue

        model_name = filename.replace("DOCLAYNET_evaluation_", "").replace("_ALL.csv", "")

        for _, row in df.iterrows():
            if row["Image"] == "AVERAGE":
                continue
            model_scores[model_name]["precision"].append(row["Precision"])
            model_scores[model_name]["recall"].append(row["Recall"])
            model_scores[model_name]["f1"].append(row["F1-Score"])
            label = row["Label"] if "Label" in row else "ALL"
            label_scores[label][model_name]["precision"].append(row["Precision"])
            label_scores[label][model_name]["recall"].append(row["Recall"])
            label_scores[label][model_name]["f1"].append(row["F1-Score"])

    results = []
    for model, scores in model_scores.items():
        avg_precision = sum(scores["precision"]) / len(scores["precision"])
        avg_recall = sum(scores["recall"]) / len(scores["recall"])
        avg_f1 = sum(scores["f1"]) / len(scores["f1"])
        results.append({
            "Model": model,
            "IOU": iou_threshold,
            "mAP": round(avg_precision, 4),
            "mAR": round(avg_recall, 4),
            "F1": round(avg_f1, 4)
        })

    comparison_path = os.path.join(base_output_folder, f"__comparison_IOU{iou_threshold}.csv")
    result_df = pd.DataFrame(results)
    result_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved in: {comparison_path}")

    label_rows = []
    for label, models in label_scores.items():
        for model, scores in models.items():
            avg_precision = sum(scores["precision"]) / len(scores["precision"])
            avg_recall = sum(scores["recall"]) / len(scores["recall"])
            avg_f1 = sum(scores["f1"]) / len(scores["f1"])
            label_rows.append({
                "category": label,
                "source": model,
                "Precision": round(avg_precision, 4),
                "Recall": round(avg_recall, 4),
                "F1-Score": round(avg_f1, 4),
                "IOU": iou_threshold
            })

    label_df = pd.DataFrame(label_rows)
    label_plot_path = os.path.join(base_output_folder, f"__category_summary_IOU{iou_threshold}.csv")
    label_df.to_csv(label_plot_path, index=False)
    print(f"Category comparison saved in: {label_plot_path}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "summary_across_IOUs.csv")

    if os.path.exists(summary_path):
        df_existing = pd.read_csv(summary_path)
        if "IOU" not in df_existing.columns:
            print("Warning: Existing summary_across_IOUs.csv does not have an “IOU” column. Create new one.")
            df_all = result_df.copy()
        else:
            df_existing = df_existing[~df_existing.set_index(["Model", "IOU"]).index.isin(
                result_df.set_index(["Model", "IOU"]).index)]
            df_all = pd.concat([df_existing, result_df], ignore_index=True)
    else:
        df_all = result_df.copy()

    df_all = df_all.sort_values(["Model", "IOU"]).reset_index(drop=True)
    df_all.to_csv(summary_path, index=False)
    print(f"\nSummary of all IOUs stored in: {summary_path}")
