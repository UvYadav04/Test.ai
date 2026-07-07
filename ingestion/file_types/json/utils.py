import json

import pandas as pd


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_records(data) -> list:
    if isinstance(data, list):
        return data
    return [data]


def flatten_records(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def infer_dtypes(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def stringify_lists(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return df
