import os
import pandas as pd

input_directory = "./Evaluator_DocLayNet/IOU_90"  # path to directory with CSVs
output_csv = "./Evaluator_DocLayNet/IOU_90/summary_metrics.csv"  # output path


def extract_metrics_from_csvs():
    summary_data = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv") and "evaluation" in filename and not filename.startswith("__"):
            filepath = os.path.join(input_directory, filename)
            try:
                df = pd.read_csv(filepath)
                if df.shape[0] > 0 and df.shape[1] >= 3:
                    last_row = df.iloc[-1]
                    metrics = last_row[-3:].tolist()
                    summary_data.append([filename] + metrics)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    summary_df = pd.DataFrame(summary_data, columns=["Filename", "Precision", "Recall", "F1"])
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary successfully written to:: {output_csv}")


if __name__ == "__main__":
    extract_metrics_from_csvs()
