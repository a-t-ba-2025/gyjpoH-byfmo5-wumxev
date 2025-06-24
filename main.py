from Evaluator_DocLayNet.evaluator import evaluate_doclaynet
from Evaluator_DocLayNet.plot_summary import plot_summary_doclaynet, plot_summary_across_ious, plot_per_label_comparison
from Evaluator_DocLayNet.summary import summarize_doclaynet

iou_threshold = 0.5


def evaluate_doc_lay_net():
    print("evaluating DocLayNet")

    for iou_threshold in [0.5, 0.75, 0.9]:
        iou_int = int(iou_threshold * 100)
        print(f"\n--- Start evaluation for IOU={iou_threshold:.2f} ---\n")
        evaluate_doclaynet(iou_threshold)  # Precision, Recall, F1 etc.
        summarize_doclaynet(iou_int)  # mAP, mAR, F1 (mean)
        plot_summary_doclaynet(iou_int)
    plot_summary_across_ious()
    plot_per_label_comparison()  #


if __name__ == "__main__":
    evaluate_doc_lay_net()
