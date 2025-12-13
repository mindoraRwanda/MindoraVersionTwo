import json
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_metrics(results):
    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["predicted_label"] for r in results]
    
    # Get unique labels
    labels = sorted(list(set(true_labels + pred_labels)))
    
    # Confusion Matrix
    cm = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        cm[(t, p)] += 1
        
    # Metrics per class
    class_metrics = {}
    for label in labels:
        tp = cm[(label, label)]
        fp = sum(cm[(l, label)] for l in labels if l != label)
        fn = sum(cm[(label, l)] for l in labels if l != label)
        tn = sum(cm[(l, p)] for l in labels for p in labels if l != label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in true_labels if t == label)
        }
        
    # Overall Accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(results) if results else 0
    
    # Macro Average
    macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(labels) if labels else 0
    macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(labels) if labels else 0
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(labels) if labels else 0
    
    # Binary Crisis Metrics (Crisis vs Non-Crisis)
    binary_true = [1 if t != "none" else 0 for t in true_labels]
    binary_pred = [1 if p != "none" else 0 for p in pred_labels]
    
    bin_tp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 1)
    bin_fp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 1)
    bin_fn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 0)
    bin_tn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 0)
    
    bin_precision = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0
    bin_recall = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0
    bin_f1 = 2 * (bin_precision * bin_recall) / (bin_precision + bin_recall) if (bin_precision + bin_recall) > 0 else 0
    bin_accuracy = (bin_tp + bin_tn) / len(results) if results else 0
    
    return {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "labels": labels,
        "binary_metrics": {
            "accuracy": bin_accuracy,
            "precision": bin_precision,
            "recall": bin_recall,
            "f1": bin_f1,
            "tp": bin_tp, "fp": bin_fp, "fn": bin_fn, "tn": bin_tn
        }
    }

def generate_visualizations(results, metrics, output_dir):
    """Generate and save visualizations using matplotlib and seaborn."""
    labels = metrics["labels"]
    
    # 1. Confusion Matrix Heatmap
    cm_data = []
    for t in labels:
        row = []
        for p in labels:
            row.append(metrics["confusion_matrix"][(t, p)])
        cm_data.append(row)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Reds', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Crisis Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_confusion_matrix.png'))
    plt.close()
    
    # 2. Metrics Bar Chart
    metrics_df = pd.DataFrame(metrics["class_metrics"]).T
    metrics_df = metrics_df[['precision', 'recall', 'f1']]
    
    metrics_df.plot(kind='bar', figsize=(14, 7))
    plt.title('Crisis Detection Metrics per Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_metrics_chart.png'))
    plt.close()

    # 3. Confidence Analysis (Box Plot: Correct vs Incorrect)
    df = pd.DataFrame(results)
    df['is_correct'] = df['true_label'] == df['predicted_label']
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_correct', y='confidence', data=df)
    plt.title('Confidence Score Distribution: Correct vs Incorrect')
    plt.xlabel('Is Correct?')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_confidence_boxplot.png'))
    plt.close()
    
    # 4. Confidence Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='confidence', hue='is_correct', multiple='stack', bins=20)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_confidence_hist.png'))
    plt.close()
    
    # 5. Error Rate per Class
    error_rates = []
    for label in labels:
        total = metrics["class_metrics"][label]["support"]
        if total > 0:
            error_rate = 1.0 - metrics["class_metrics"][label]["f1"]
            error_rates.append({'label': label, 'error_rate': error_rate})
        else:
            error_rates.append({'label': label, 'error_rate': 0.0})
            
    err_df = pd.DataFrame(error_rates)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='label', y='error_rate', data=err_df, palette='Reds')
    plt.title('Model Difficulty per Class (1 - F1 Score)')
    plt.ylabel('Error Score (1 - F1)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_error_rate.png'))
    plt.close()

    # 6. Severity Distribution
    severity_counts = df['severity'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Predicted Severity Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_severity_dist.png'))
    plt.close()

    # 7. Text Length vs Confidence
    df['text_length'] = df['text'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='text_length', y='confidence', hue='is_correct', style='is_correct')
    plt.title('Text Length vs Confidence Score')
    plt.xlabel('Text Length (chars)')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crisis_text_length_scatter.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_report(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
        
    metrics = calculate_metrics(results)
    output_dir = os.path.dirname(results_path)
    
    # Generate Visualizations
    generate_visualizations(results, metrics, output_dir)
    
    report = []
    report.append("# Crisis Detection Experiment Report\n")
    
    report.append("## 1. Binary Crisis Detection (Crisis vs None)")
    report.append(f"- **Accuracy**: {metrics['binary_metrics']['accuracy']:.4f}")
    report.append(f"- **Precision**: {metrics['binary_metrics']['precision']:.4f}")
    report.append(f"- **Recall**: {metrics['binary_metrics']['recall']:.4f}")
    report.append(f"- **F1-Score**: {metrics['binary_metrics']['f1']:.4f}")
    report.append(f"- **Confusion Matrix**: TP={metrics['binary_metrics']['tp']}, FP={metrics['binary_metrics']['fp']}, FN={metrics['binary_metrics']['fn']}, TN={metrics['binary_metrics']['tn']}\n")
    
    report.append("## 2. Detailed Class-wise Metrics")
    report.append(f"- **Overall Accuracy**: {metrics['accuracy']:.4f}")
    report.append(f"- **Macro F1-Score**: {metrics['macro_avg']['f1']:.4f}\n")
    
    report.append("| Label | Precision | Recall | F1-Score | Support |")
    report.append("|---|---|---|---|---|")
    for label in metrics["labels"]:
        m = metrics["class_metrics"][label]
        report.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    report.append("\n")
    
    report.append("## 3. Confusion Matrix")
    report.append("![Confusion Matrix](crisis_confusion_matrix.png)\n")
    # Header
    header = "| True \\ Pred | " + " | ".join(metrics["labels"]) + " |"
    report.append(header)
    report.append("|---" + "|---" * len(metrics["labels"]) + "|")
    
    for t in metrics["labels"]:
        row = f"| **{t}** |"
        for p in metrics["labels"]:
            row += f" {metrics['confusion_matrix'][(t, p)]} |"
        report.append(row)
    report.append("\n")
    
    report.append("## 4. Visualizations")
    report.append("### Metrics")
    report.append("![Metrics Chart](crisis_metrics_chart.png)\n")
    report.append("### Confidence Analysis")
    report.append("![Confidence Boxplot](crisis_confidence_boxplot.png)")
    report.append("![Confidence Histogram](crisis_confidence_hist.png)\n")
    report.append("### Error Analysis")
    report.append("![Error Rate](crisis_error_rate.png)\n")
    report.append("### Severity Analysis")
    report.append("![Severity Distribution](crisis_severity_dist.png)")
    report.append("![Text Length vs Confidence](crisis_text_length_scatter.png)\n")
    
    report.append("## 5. Error Analysis")
    report.append("### Misclassified Examples")
    
    errors = [r for r in results if r["true_label"] != r["predicted_label"]]
    if not errors:
        report.append("No errors found.")
    else:
        for i, err in enumerate(errors[:10]): # Show top 10 errors
            report.append(f"**{i+1}. Text**: \"{err['text']}\"")
            report.append(f"   - **True**: {err['true_label']}")
            report.append(f"   - **Predicted**: {err['predicted_label']}")
            report.append(f"   - **Severity**: {err['severity']}")
            report.append(f"   - **Confidence**: {err['confidence']:.4f}")
            report.append("")
            
    print("\n".join(report))
    
    # Save report
    output_path = os.path.join(output_dir, "crisis_report.md")
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    results_file = os.path.join(os.path.dirname(__file__), "crisis_results.json")
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}. Run experiments first.")
    else:
        generate_report(results_file)
