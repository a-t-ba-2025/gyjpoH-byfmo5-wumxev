import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_summary_doclaynet(iou_threshold, input_dir="Evaluator_DocLayNet", output_dir="Evaluator_DocLayNet/plots"):
    iou_str = f"IOU_{int(iou_threshold):02d}"
    summary_file = os.path.join(input_dir, iou_str, f"__comparison_IOU{int(iou_threshold):02d}.csv")

    if not os.path.exists(summary_file):
        print(f"File not found: {summary_file}")
        return

    df = pd.read_csv(summary_file)
    output_path = os.path.join(output_dir, iou_str)
    os.makedirs(output_path, exist_ok=True)

    def make_single_bar_plot(metric: str, ylabel: str):
        models = df["Model"].tolist()
        values = df[metric].tolist()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values)
        plt.ylabel(ylabel)
        plt.title(f"{metric} bei IOU {iou_threshold/100:.2f}")
        plt.ylim(0, 1.0)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f'{val:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        filename = os.path.join(output_path, f"{metric.lower()}_IOU{int(iou_threshold):02d}.png")
        plt.savefig(filename)
        plt.close()

    for metric in ["mAP", "mAR", "F1"]:
        make_single_bar_plot(metric, metric)

    print(f"Plots saved in: {output_path}")


def plot_summary_across_ious(input_csv="Evaluator_DocLayNet/summary_across_IOUs.csv", output_dir="Evaluator_DocLayNet/plots/across_IOUs"):
    if not os.path.exists(input_csv):
        print(f"Datei nicht gefunden: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["mAP", "mAR", "F1"]
    models = df["Model"].unique()
    ious = sorted(df["IOU"].unique())

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for model in models:
            model_data = df[df["Model"] == model].sort_values("IOU")
            plt.plot(model_data["IOU"], model_data[metric], marker="o", label=model)

        plt.xlabel("IOU Threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} über verschiedene IOU-Stufen")
        plt.ylim(0, 1.0)
        plt.xticks(ious)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric}_across_IOUs.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Plot gespeichert: {output_path}")

def plot_per_label_comparison(input_dir="Evaluator_DocLayNet", output_dir="Evaluator_DocLayNet/plots/per_label"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["Precision", "Recall", "F1-Score"]
    iou_folders = [f for f in os.listdir(input_dir) if f.startswith("IOU_")]

    for iou_folder in iou_folders:
        folder_path = os.path.join(input_dir, iou_folder)
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and not f.startswith("__")]

        label_data = []

        for file in all_files:
            filepath = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(filepath)
                avg_row = df[df["Image"] == "AVERAGE"].iloc[0]

                precision = avg_row.get("Precision", 0.0)
                recall = avg_row.get("Recall", 0.0)
                f1 = avg_row.get("F1-Score", 0.0)

                filename = file.replace("DOCLAYNET_evaluation_", "").replace(".csv", "")
                parts = filename.split("_")
                model = parts[0] if len(parts) > 0 else "UNKNOWN"
                label = parts[1] if len(parts) > 1 else "ALL"

                label_data.append({
                    "IOU": iou_folder.replace("IOU_", ""),
                    "Model": model,
                    "Label": label,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")

        if not label_data:
            continue

        df_plot = pd.DataFrame(label_data)

        for metric in metrics:
            plt.figure(figsize=(14, 6))
            labels = sorted(df_plot["Label"].unique())
            models = sorted(df_plot["Model"].unique())
            x = np.arange(len(labels))
            width = 0.25

            for i, model in enumerate(models):
                vals = []
                for label in labels:
                    value = df_plot[(df_plot["Model"] == model) & (df_plot["Label"] == label)][metric]
                    vals.append(value.values[0] if not value.empty else 0.0)
                plt.bar(x + i * width, vals, width=width, label=model)

            plt.xticks(x + width * (len(models)-1)/2, labels, rotation=45)
            plt.ylim(0, 1.0)
            plt.title(f"{metric} pro Label – IOU {iou_folder.replace('IOU_', '')}")
            plt.xlabel("Label")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"{metric}_bars_{iou_folder}.png")
            plt.savefig(out_path)
            plt.close()
            print(f"Plot saved: {out_path}")
