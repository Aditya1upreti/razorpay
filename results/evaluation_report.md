# Sentinel — Evaluation Report

## Classification Metrics

| Metric | Value |
|--------|-------|
| Precision | 1.0 |
| Recall | 0.4 |
| F1 Score | 0.5714 |
| False Positive Rate | 0.0 |
| False Negative Rate | 0.6 |
| TP / FP / TN / FN | 2 / 0 / 81 / 3 |
| Manual Review Count | 10 (10.4%) |
| Evaluated (excl. manual_review) | 86 / 96 |

## Ring Detection Accuracy

| Metric | Value |
|--------|-------|
| Precision | 1.0 |
| Recall | 1.0 |
| F1 Score | 1.0 |
| Detected Ring Accounts | 3 |
| Ground Truth Ring Accounts (in test set) | 3 |
| True Positives / False Positives / False Negatives | 3 / 0 / 0 |

## Revenue Recovery

| Metric | Value |
|--------|-------|
| Total At Risk | INR 15,283.11 |
| Total Recovered | INR 15,283.11 |
| Recovery Rate | 100.0% |
| Soft Decline Recovered | INR 8,023.28 |
| Hard Decline Recovered | INR 7,259.83 |
| Suppressed Count | 0 |

## Confidence Calibration Table

| Bucket | Count | Actual Accuracy |
|--------|-------|-----------------|
| 0-20 | 0 | nan% |
| 20-40 | 0 | nan% |
| 40-60 | 10 | 0.0% |
| 60-80 | 1 | 100.0% |
| 80-100 | 0 | nan% |

### Calibration Sample-Size Warnings

- Bucket '60-80' has only 1 samples — calibration estimate has low statistical confidence.
