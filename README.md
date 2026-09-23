# Transformer + Isolation Forest Anomaly Detection Reproduction

This project reproduces the model described in `AI驱动模型在异常检测中的应用研究.docx`.

## Data Layout

Put local datasets here:

- `data/raw/fall/` for UCI Localization / fall detection files
- `data/raw/cicids2017/` for CIC-IDS2017 subset files

Supported file formats: `.csv`, `.xlsx`, `.xls`, `.parquet`.

The loader tries common label names automatically: `label`, `Label`, `class`, `Class`, `target`, `Target`, `activity`.
Normal labels are inferred from `BENIGN`, `normal`, `0`, `false`, and `negative`; other labels are treated as anomalies.

## Commands

Run a synthetic end-to-end check:

```bash
python -m src.smoke_test
```

Run the complete thesis experiment suite with baselines and ablations:

```bash
python -m src.experiment --dataset synthetic --fast
python -m src.experiment --dataset fall --data-path data/raw/fall
python -m src.experiment --dataset cicids --data-path data/raw/cicids2017
```

Each experiment writes `summary.csv`, `report.md`, per-method metrics, PR curves, confusion matrices, and training loss plots under `outputs/experiment_*`.

Train fall detection:

```bash
python -m src.train --dataset fall --data-path data/raw/fall
```

Train CIC-IDS2017:

```bash
python -m src.train --dataset cicids --data-path data/raw/cicids2017
```

Evaluate an existing run:

```bash
python -m src.evaluate --run-dir outputs/<run-name>
```

## Notes

The paper contains conflicting reported metrics across sections. This code records actual metrics from the run and does not force results to match a table.

Optuna is optional. If it is unavailable, the trainer uses a deterministic compact search.
