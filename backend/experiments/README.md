# Experiments

This directory contains scripts and datasets for running and analyzing experiments on the Emotion Classifier and Crisis Detector.

## Prerequisites

Ensure you have the necessary environment variables set up in `../../.env.development`. The scripts automatically load this file.

You need to have the backend dependencies installed. If you are running this from the root of the project, you might need to activate your virtual environment.

## 1. Emotion Classification Experiments

This experiment evaluates the performance of the emotion classifier against a labeled dataset.

### Running the Experiment

To run the emotion classification experiment:

```bash
python run_experiments.py
```

- **Input**: `dataset.json` (List of text examples with true labels)
- **Output**: `results.json` (Contains predictions, confidence scores, and full responses)

### Analyzing Results

To generate a report and visualizations from the results:

```bash
python analyze_results.py
```

- **Input**: `results.json`
- **Output**:
    - `report.md`: A summary report of the metrics.
    - `emotion_confusion_matrix.png`: Confusion matrix heatmap.
    - `emotion_metrics_chart.png`: Bar chart of precision, recall, and F1-score per class.
    - `emotion_confidence_boxplot.png`: Box plot of confidence scores for correct vs incorrect predictions.
    - `emotion_confidence_hist.png`: Histogram of confidence scores.
    - `emotion_error_rate.png`: Error rate per class.
    - `emotion_intensity_dist.png`: Distribution of predicted intensities.
    - `emotion_text_length_scatter.png`: Scatter plot of text length vs confidence.

## 2. Crisis Detection Experiments

This experiment evaluates the performance of the crisis detector.

### Running the Experiment

To run the crisis detection experiment:

```bash
python run_crisis_experiments.py
```

- **Input**: `crisis_dataset.json` (List of text examples with true labels)
- **Output**: `crisis_results.json` (Contains predictions, severity, confidence, and rationale)

### Analyzing Results

To generate a report and visualizations from the crisis results:

```bash
python analyze_crisis_results.py
```

- **Input**: `crisis_results.json`
- **Output**:
    - `crisis_report.md`: A summary report of the metrics.
    - `crisis_confusion_matrix.png`: Confusion matrix heatmap.
    - `crisis_metrics_chart.png`: Bar chart of metrics per class.
    - `crisis_confidence_boxplot.png`: Box plot of confidence scores.
    - `crisis_confidence_hist.png`: Histogram of confidence scores.
    - `crisis_error_rate.png`: Error rate per class.
    - `crisis_severity_dist.png`: Distribution of predicted severity.
    - `crisis_text_length_scatter.png`: Scatter plot of text length vs confidence.

## Summary of Files

- `dataset.json`: Ground truth dataset for emotion classification.
- `crisis_dataset.json`: Ground truth dataset for crisis detection.
- `run_experiments.py`: Script to run emotion classification inference.
- `run_crisis_experiments.py`: Script to run crisis detection inference.
- `analyze_results.py`: Script to analyze emotion classification results.
- `analyze_crisis_results.py`: Script to analyze crisis detection results.
