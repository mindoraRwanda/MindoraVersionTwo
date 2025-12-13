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
    
    return {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "class_metrics": class_metrics,
        "confusion_matrix": cm,
        "labels": labels
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
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Emotion Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_confusion_matrix.png'))
    plt.close()
    
    # 2. Metrics Bar Chart
    metrics_df = pd.DataFrame(metrics["class_metrics"]).T
    metrics_df = metrics_df[['precision', 'recall', 'f1']]
    
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Emotion Classification Metrics per Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_metrics_chart.png'))
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
    plt.savefig(os.path.join(output_dir, 'emotion_confidence_boxplot.png'))
    plt.close()
    
    # 4. Confidence Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='confidence', hue='is_correct', multiple='stack', bins=20)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_confidence_hist.png'))
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
    plt.savefig(os.path.join(output_dir, 'emotion_error_rate.png'))
    plt.close()

    # 6. Intensity Distribution
    # Extract intensity from full_response if available
    if 'full_response' in df.columns:
        df['intensity'] = df['full_response'].apply(lambda x: x.get('intensity', 'unknown') if isinstance(x, dict) else 'unknown')
        intensity_counts = df['intensity'].value_counts()
        
        plt.figure(figsize=(8, 8))
        plt.pie(intensity_counts, labels=intensity_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Emotion Intensity Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_intensity_dist.png'))
        plt.close()

    # 7. Text Length vs Confidence
    df['text_length'] = df['text'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='text_length', y='confidence', hue='is_correct', style='is_correct')
    plt.title('Text Length vs Confidence Score')
    plt.xlabel('Text Length (chars)')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_text_length_scatter.png'))
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
    report.append("# Capstone Project Experiment Report\n")
    
    report.append("## 1. Experimental Results Summary")
    report.append(f"- **Total Samples**: {len(results)}")
    report.append(f"- **Accuracy**: {metrics['accuracy']:.4f}")
    report.append(f"- **Macro Precision**: {metrics['macro_avg']['precision']:.4f}")
    report.append(f"- **Macro Recall**: {metrics['macro_avg']['recall']:.4f}")
    report.append(f"- **Macro F1-Score**: {metrics['macro_avg']['f1']:.4f}\n")
    
    report.append("## 2. Class-wise Metrics")
    report.append("| Label | Precision | Recall | F1-Score | Support |")
    report.append("|---|---|---|---|---|")
    for label in metrics["labels"]:
        m = metrics["class_metrics"][label]
        report.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    report.append("\n")
    
    report.append("## 3. Confusion Matrix")
    report.append("![Confusion Matrix](emotion_confusion_matrix.png)\n")
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
    report.append("![Metrics Chart](emotion_metrics_chart.png)\n")
    report.append("### Confidence Analysis")
    report.append("![Confidence Boxplot](emotion_confidence_boxplot.png)")
    report.append("![Confidence Histogram](emotion_confidence_hist.png)\n")
    report.append("### Error Analysis")
    report.append("![Error Rate](emotion_error_rate.png)\n")
    report.append("### Additional Analysis")
    report.append("![Intensity Distribution](emotion_intensity_dist.png)")
    report.append("![Text Length vs Confidence](emotion_text_length_scatter.png)\n")
    
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
            report.append(f"   - **Confidence**: {err['confidence']:.4f}")
            report.append("")
            
    print("\n".join(report))
    
    # Save report
    output_path = os.path.join(output_dir, "report.md")
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    results_file = os.path.join(os.path.dirname(__file__), "results.json")
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}. Run experiments first.")
    else:
        generate_report(results_file)
