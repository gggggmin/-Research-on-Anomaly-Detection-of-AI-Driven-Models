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

## Fast Reproduction Results

These results use `--fast`, so they are suitable as a quick local reproduction and sanity check. For final thesis tables, rerun without `--fast`.

### Fall Detection

Command:

```bash
python -m src.experiment --dataset fall --data-path data/raw/fall --fast
```

Output directory:

`outputs/experiment_fall_20260923_203809`

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| full_transformer_iforest | 0.0334 | 0.0162 | 1.0000 | 0.0319 | 0.3497 |
| zscore | 0.2403 | 0.0186 | 0.9048 | 0.0365 | 0.4936 |
| iforest_without_transformer | 0.0212 | 0.0160 | 1.0000 | 0.0315 | 0.2576 |
| one_class_svm | 0.0334 | 0.0162 | 1.0000 | 0.0319 | 0.2633 |
| transformer_without_iforest | 0.6073 | 0.0212 | 0.5238 | 0.0407 | 0.5194 |
| autoencoder_iforest | 0.0629 | 0.0167 | 1.0000 | 0.0329 | 0.4462 |

### CIC-IDS2017 DDoS Intrusion Detection

Command:

```bash
python -m src.experiment --dataset cicids --data-path data/raw/cicids2017 --feature-mode top7 --fast
```

Output directory:

`outputs/experiment_cicids_20260923_203825`

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| full_transformer_iforest | 0.7686 | 0.7105 | 0.9989 | 0.8304 | 0.8058 |
| zscore | 0.7835 | 0.9909 | 0.6240 | 0.7658 | 0.6955 |
| iforest_without_transformer | 0.7459 | 0.6910 | 0.9983 | 0.8167 | 0.7661 |
| one_class_svm | 0.9337 | 0.9004 | 0.9928 | 0.9444 | 0.9373 |
| transformer_without_iforest | 0.5672 | 0.5672 | 1.0000 | 0.7238 | 0.3919 |
| autoencoder_iforest | 0.7170 | 0.6675 | 0.9983 | 0.8001 | 0.7548 |

## Interpretation

The CIC-IDS2017 DDoS subset produces usable intrusion-detection evidence, but the current fast run does not reproduce the paper's claimed `0.92` accuracy for the fusion model. One-Class SVM is strongest in this quick setting.

The fall experiment is not thesis-grade evidence yet. The UCI Localization dataset is a localization/activity dataset with only `falling` as a small minority class, not the simplified 6,624-row accelerometer dataset described in the thesis text. The thesis likely used an undocumented subset, filtering strategy, or different fall dataset. Treating the full UCI Localization file directly as a binary fall dataset gives weak metrics.

For final reporting, run non-fast experiments after deciding whether to:

- keep the full UCI Localization data and honestly report weak fall performance;
- filter to specific subjects/tags/activities and document that filtering;
- replace the fall dataset with a true fall-detection accelerometer dataset that matches the thesis claim.
