from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UCI_LOCALIZATION_COLUMNS = [
    "sequence",
    "tag_id",
    "timestamp",
    "date",
    "x",
    "y",
    "z",
    "Label",
]


def prepare_uci_localization(raw_path: str = "data/raw/fall/ConfLongDemo_JSI.txt") -> Path:
    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(f"Missing UCI Localization source file: {source}")
    out_path = source.with_name("uci_localization_fall.csv")
    df = pd.read_csv(source, header=None, names=UCI_LOCALIZATION_COLUMNS)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows.")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare downloaded public datasets for experiments.")
    parser.add_argument("--uci-localization", default="data/raw/fall/ConfLongDemo_JSI.txt")
    args = parser.parse_args()
    prepare_uci_localization(args.uci_localization)


if __name__ == "__main__":
    main()
