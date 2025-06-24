
## Evaluation (DocLayNet Evaluator)

Evaluates results of self trained layout detection models (e.g. DETR, Faster R-CNN) on the DocLayNet dataset. It computes metrics like Precision, Recall, F1-Score as well as mAP/mAR per class, model, and IOU level.

---

### Structure

```
├── main.py
├── extract_metrics_from_csv.py    # Extracts metrics summary from CSVs
├── Evaluator_DocLayNet/
    ├── evaluator.py               # Runs category-wise and global evaluation
    ├── summary.py                 # Aggregates metrics across IOU levels
    ├── plot_summary.py            # Creates visualizations for metrics


```

---

### Configuration

The `.env` file must contain the following variables:

```ini
# ADD PATH TO GROUNDTRUTH (for Example /Datasets/DocLayNet_core/COCO/test.json")
GT_JSON= 
# ADD PATH TO PREDICTIONS
PRED_FOLDER= 
```

- `GT_JSON`: Path to the COCO-style test file (`test.json`) from DocLayNet  
- `PRED_FOLDER`: Directory with prediction `.json` files for each image  

Ensure predicted files are named identically to the original images (e.g., `xxx.json` ⇨ `xxx.png`).

---

### Run Evaluation

Execute the evaluation by running:

```bash
python main.py
```

This will:
- Evaluate each layout class separately (e.g. Table, Section-header, Title, ...)
- Compute metrics for three IOU thresholds: **0.50**, **0.75**, **0.90**
- Save detailed results in CSV format and generate summary plots


Visualizations are saved in:

```
Evaluator_DocLayNet/plots/
```

---

### Output Overview

- **Per-model CSVs**: Precision, Recall, F1-Score (per image + average)
- **Summary CSVs**: Mean values for each model and IOU
- **Plots**: 
  - Bar charts per model and IOU (`plots/`)
  - Label-wise comparisons (`plots/per_label/`)
  - IOU performance (`plots/across_IOUs/`)

---

### Extract Summary Table Only

If you only want to extract the final scores across all classes for a given IOU:

```bash
python Evaluator_DocLayNet/extract_metrics_from_csv.py
```

---

### Install requirements

```bash
pip install -r requirements.txt
```

---
