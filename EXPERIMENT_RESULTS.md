# Experiment Results

## Downloaded Data

- Fall detection: UCI Localization Data for Person Activity, saved under `data/raw/fall/`.
- Network intrusion: CIC-IDS2017 machine-learning DDoS subset, saved under `data/raw/cicids2017/`.

The raw datasets are intentionally ignored by Git because they are generated/downloaded data.

## Data Checks

| Dataset | Local file | Rows | Features/columns | Label distribution |
|---|---|---:|---:|---|
| UCI Localization | `data/raw/fall/uci_localization_fall.csv` | 164,860 | 8 raw columns | `falling`: 2,973; other activities: 161,887 |
| CIC-IDS2017 DDoS subset | `data/raw/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet` | 225,745 | 79 columns | `DDoS`: 128,027; `BENIGN`: 97,718 |

After grouping UCI Localization by `(sequence, tag_id)` and applying `window=50`, `stride=25`, the fall experiment contains:

| Split | Normal windows | Fall windows |
|---|---:|---:|
| Train | 4,590 | 566 |
| Test | 1,148 | 141 |

## Fast Reproduction Results

These results use `--fast`, so they are suitable as a quick local reproduction and sanity check. For final thesis tables, rerun without `--fast`.

The optimized suite includes a supervised score-fusion variant:

`supervised_transformer_iforest_fusion = 0.65 * Transformer classification score + 0.35 * IsolationForest score`

This variant uses labels during Transformer training, so it should be described as supervised or weakly supervised representation learning, not as a purely unsupervised detector.

### Fall Detection

Command:

```bash
python -m src.experiment --dataset fall --data-path data/raw/fall --fast
```

Output directory:

`outputs/experiment_fall_20260923_205441`

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| full_transformer_iforest | 0.1094 | 0.1094 | 1.0000 | 0.1972 | 0.5075 |
| supervised_transformer_iforest_fusion | 0.9193 | 0.6069 | 0.7447 | 0.6688 | 0.9259 |
| zscore | 0.1629 | 0.1137 | 0.9787 | 0.2037 | 0.5052 |
| iforest_without_transformer | 0.1109 | 0.1096 | 1.0000 | 0.1975 | 0.4939 |
| one_class_svm | 0.1955 | 0.1138 | 0.9362 | 0.2029 | 0.4874 |
| transformer_without_iforest | 0.6245 | 0.1478 | 0.5106 | 0.2293 | 0.5972 |
| autoencoder_iforest | 0.3274 | 0.1227 | 0.8369 | 0.2140 | 0.5754 |

### CIC-IDS2017 DDoS Intrusion Detection

Command:

```bash
python -m src.experiment --dataset cicids --data-path data/raw/cicids2017 --feature-mode top7 --fast
```

Output directory:

`outputs/experiment_cicids_20260923_205321`

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| full_transformer_iforest | 0.7743 | 0.7157 | 0.9987 | 0.8339 | 0.8015 |
| supervised_transformer_iforest_fusion | 0.9979 | 0.9988 | 0.9976 | 0.9982 | 0.9992 |
| zscore | 0.7835 | 0.9909 | 0.6240 | 0.7658 | 0.6960 |
| iforest_without_transformer | 0.7588 | 0.7021 | 0.9983 | 0.8244 | 0.7798 |
| one_class_svm | 0.9221 | 0.8841 | 0.9928 | 0.9353 | 0.9280 |
| transformer_without_iforest | 0.6570 | 0.6231 | 1.0000 | 0.7678 | 0.6926 |
| autoencoder_iforest | 0.7124 | 0.6638 | 0.9987 | 0.7975 | 0.7552 |

## Interpretation

The CIC-IDS2017 DDoS subset produces strong evidence for the optimized supervised fusion model. It outperforms One-Class SVM and the unsupervised Transformer-IsolationForest variant in this quick run.

The fall experiment is improved by grouped windowing, event-style labels, and supervised score fusion. Even so, UCI Localization should still be described carefully because it is a localization/activity dataset, not the simplified 6,624-row accelerometer dataset described in the original thesis text.

The strongest defensible claim is:

> A supervised Transformer representation learner combined with IsolationForest score fusion performs strongly on both the CIC-IDS2017 DDoS subset and the binary UCI Localization fall-vs-nonfall task.

For final reporting, run non-fast experiments after deciding whether to:

- keep the full UCI Localization data and honestly report weak fall performance;
- filter to specific subjects/tags/activities and document that filtering;
- replace the fall dataset with a true fall-detection accelerometer dataset that matches the thesis claim.
