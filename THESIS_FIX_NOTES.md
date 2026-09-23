# Thesis Experiment Fix Notes

## Blunt Assessment

The original thesis experiment is not fully defensible as written. The high-level idea, "Transformer feature extractor + Isolation Forest anomaly detector", is reasonable, but the experimental description mixes incompatible assumptions:

- UCI Localization is not a simple 6,624-row three-axis fall dataset. The official file has 164,860 rows and four body tags mixed across five subjects.
- CIC-IDS2017 DDoS subset has 225,745 rows in the downloaded machine-learning version, close to the thesis number, but its anomaly ratio is about 56.7%, not 38%.
- Some reported metrics conflict across sections, especially network intrusion accuracy and the ablation table.
- Isolation Forest should be trained on normal samples for anomaly detection. Training it on a mixed set with majority attacks makes the score semantics collapse.

So the paper should be repaired, not cosmetically defended.

## Corrected Experimental Design

Use the following defensible setup:

1. Data source disclosure
   - Fall: UCI Localization Data for Person Activity, binary label `falling` vs non-falling activities.
   - Network: CIC-IDS2017 machine-learning DDoS subset, binary label `DDoS` vs `BENIGN`.

2. Fall preprocessing
   - Sort and window data within each `(sequence, tag_id)` group.
   - Do not create sliding windows across different body tags.
   - Use `window=50`, `stride=25`.
   - Extract x/y/z, total movement magnitude, tilt proxy, local means, and local standard deviations.
   - Mark a window as abnormal if any row inside the window is labelled `falling`, because fall detection is an event detection task.

3. Network preprocessing
   - Replace infinities with missing values.
   - Fill missing values with medians.
   - Remove features with Pearson correlation above `0.9`.
   - Select top features using RandomForest importance.

4. Training protocol
   - Split data with stratification.
   - Train anomaly detectors only on normal samples from the training set.
   - Evaluate on the full test set.
   - Report Accuracy, Precision, Recall, F1, ROC-AUC, PR curve, confusion matrix.

5. Baselines and ablations
   - Z-Score
   - One-Class SVM
   - raw Isolation Forest
   - Transformer reconstruction score without Isolation Forest
   - AutoEncoder + Isolation Forest
   - Transformer + Isolation Forest

## Suggested Rewrite

Replace exaggerated claims such as "the fusion model significantly outperforms all baselines in every scenario" with a more honest conclusion:

> The Transformer-Isolation Forest fusion model improves feature representation for high-dimensional intrusion data and performs competitively on CIC-IDS2017. However, on UCI Localization fall detection, performance is limited by class imbalance and the fact that the dataset records localization coordinates rather than dedicated acceleration signals. This shows that the fusion framework is promising for network anomaly detection, while fall detection requires either stricter subject/tag filtering, stronger imbalance handling, or a more suitable sensor fall dataset.

This is less flashy, but it will survive questioning.
