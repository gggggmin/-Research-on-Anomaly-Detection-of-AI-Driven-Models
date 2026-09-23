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
   - For the optimized model, train the Transformer with a weighted binary classification objective plus reconstruction loss, then fuse Transformer probability with Isolation Forest anomaly score.
   - Evaluate on the full test set.
   - Report Accuracy, Precision, Recall, F1, ROC-AUC, PR curve, confusion matrix.

5. Baselines and ablations
   - Z-Score
   - One-Class SVM
   - raw Isolation Forest
   - Transformer reconstruction score without Isolation Forest
   - AutoEncoder + Isolation Forest
   - Transformer + Isolation Forest
   - Supervised Transformer + Isolation Forest score fusion

## Suggested Rewrite

Replace exaggerated claims such as "the unsupervised fusion model significantly outperforms all baselines in every scenario" with a more accurate conclusion:

> The proposed optimized model uses Transformer-based supervised representation learning and Isolation Forest score fusion. Experiments show that the optimized fusion model achieves strong performance on both CIC-IDS2017 DDoS intrusion detection and UCI Localization fall-vs-nonfall detection. Ablation results indicate that using Isolation Forest without Transformer features performs worse, and replacing score fusion with single-module decisions reduces F1 and AUC. These results support the effectiveness of combining Transformer representation learning with isolation-based anomaly scoring, while the method should be described as supervised or weakly supervised rather than purely unsupervised.

This is less flashy, but it will survive questioning.
