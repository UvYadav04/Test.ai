import csv

import pandas as pd


def detect_delimiter(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ","


def infer_dtypes(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}
