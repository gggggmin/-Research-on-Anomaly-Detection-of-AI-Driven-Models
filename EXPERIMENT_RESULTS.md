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

### Fall Detection

Command:

```bash
python -m src.experiment --dataset fall --data-path data/raw/fall --fast
```

Output directory:

`outputs/experiment_fall_20260923_204542`

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| full_transformer_iforest | 0.6043 | 0.1375 | 0.4965 | 0.2154 | 0.5561 |
| zscore | 0.3646 | 0.1182 | 0.7447 | 0.2041 | 0.5059 |
| iforest_without_transformer | 0.1389 | 0.1115 | 0.9858 | 0.2003 | 0.4603 |
| one_class_svm | 0.2040 | 0.1162 | 0.9504 | 0.2071 | 0.4968 |
| transformer_without_iforest | 0.5493 | 0.1573 | 0.7163 | 0.2580 | 0.6145 |
| autoencoder_iforest | 0.7998 | 0.2201 | 0.3262 | 0.2629 | 0.5995 |

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

The fall experiment is improved by grouped windowing and event-style labels: a window is anomalous if it contains any `falling` row. Even after this correction, the result is modest because UCI Localization is a localization/activity dataset, not the simplified 6,624-row accelerometer dataset described in the thesis text. The thesis likely used an undocumented subset, filtering strategy, or different fall dataset.

For final reporting, run non-fast experiments after deciding whether to:

- keep the full UCI Localization data and honestly report weak fall performance;
- filter to specific subjects/tags/activities and document that filtering;
- replace the fall dataset with a true fall-detection accelerometer dataset that matches the thesis claim.
