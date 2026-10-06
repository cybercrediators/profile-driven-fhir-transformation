#!/usr/bin/env python3
"""Preprocessing for the MOTU dataset

Based on data_cleaning.ipynb from the adhb-mapping-service!

Output: one JSON array per table in `_data/records/<table>.records.json`.
"""
import json
import math
import re
from functools import reduce
from os import path, makedirs

import numpy as np
import pandas as pd
from more_itertools import partition

HERE = path.dirname(path.abspath(__file__))
DATA = path.join(HERE, "_data")
CSV = path.join(DATA, "csv")
OUT = path.join(DATA, "records")

with open(path.join(DATA, "scripts/data_cleaning.whitelist.json")) as f:
    WL = json.load(f)
with open(path.join(DATA, "scripts/data_cleaning.types.json")) as f:
    TYPES = json.load(f)


def clean_patient():
    df = pd.read_csv(path.join(CSV, "Patient.csv"), sep=",", encoding="latin1")
    df.rename(columns=WL["patient"], inplace=True)
    df = df[list(WL["patient"].values())]
    df["birthday"] = pd.to_datetime(df["birthday"], format="%Y-%m-%d", errors="coerce")
    return df


def clean_hospital_stay():
    df = pd.read_csv(path.join(CSV, "HospitalStay.csv"), sep=",", encoding="latin1")
    df.rename(columns=WL["hospital_stay"], inplace=True)
    df = df[list(WL["hospital_stay"].values())]
    for column, data_type in TYPES["hospital_stay"].items():
        if column in df.columns:
            df[column] = df[column].astype(data_type)
    df["hospitalization.admissionDate"] = pd.to_datetime(
        df["hospitalization.admissionDate"], format="%Y-%m-%d"
    )
    date_columns = [x for x in df.columns if "date" in x.lower()]
    date_columns.remove("hospitalization.admissionDate")
    for column in date_columns:
        df[column] = pd.to_datetime(df[column], format="%Y-%m-%d", errors="coerce")

    # prostheticKnee composite "A_B" -> validate against Knee table, then "A;B"
    df_knee = pd.read_csv(path.join(CSV, "Knee.csv"), sep=",")
    not_matching, _ = partition(
        lambda x: x in list(df_knee["ProstheticKnee"]), df["prostheticKnee"].unique()
    )
    _wo, with_underscore = partition(
        lambda x: "_" in x, filter(lambda x: type(x) == str, not_matching)
    )
    acc = []
    for s in with_underscore:
        for model in map(lambda x: x.strip(), s.split("_")):
            if model not in acc:
                acc.append(model)
    not_matching, _matching = partition(lambda x: x in list(df_knee["ProstheticKnee"]), acc)
    not_matching = list(not_matching)
    if len(not_matching) != 0:
        raise ValueError(f"Not matching knee values should be 0: {not_matching}")

    def transform(input):
        if type(input) == str:
            return reduce(
                lambda acc, x: acc + ";" + x, map(lambda x: x.strip(), input.split("_"))
            )
        return input

    df["prostheticKnee"] = df["prostheticKnee"].map(transform)
    return df


def clean_knee():
    df = pd.read_csv(path.join(CSV, "Knee.csv"), sep=",")
    df.rename(columns=WL["knee"], inplace=True)
    df = df[list(WL["knee"].values())]
    for column, data_type in TYPES["knee"].items():
        if column in df.columns:
            df[column] = df[column].astype(data_type)
    r1 = r"([0-9]+(\.[0-9]+)?) +kg"
    r2 = r"([0-9]+(\.[0-9]+)?) *[-–] *([0-9]+(\.[0-9]+)?) +kg"

    def transform(input):
        if type(input) == str:
            m = re.match(r1, input)
            if m:
                return float(m.group(1))
            m = re.match(r2, input)
            if m:
                return (float(m.group(1)) + float(m.group(3))) / 2
        if isinstance(input, float) and math.isnan(input):
            return input

    df["weight"] = df["weight"].map(transform)
    df["patientMaximumWeight"] = df["patientMaximumWeight"].map(transform)
    return df


def clean_drug():
    df = pd.read_csv(path.join(CSV, "Drug.csv"), sep=",")
    df.rename(columns=WL["drug"], inplace=True)
    df = df[list(WL["drug"].values())]
    df["admissionDate"] = pd.to_datetime(df["admissionDate"], format="%Y-%m-%d")
    df = df[df["atc"].notna()]
    return df


def clean_fall():
    df = pd.read_csv(path.join(CSV, "Fall.csv"), sep=",")
    df.rename(columns=WL["fall"], inplace=True)
    df = df[list(WL["fall"].values())]
    for column, data_type in TYPES["fall"].items():
        if column in df.columns:
            df[column] = df[column].astype(data_type)
    df["admissionDate"] = pd.to_datetime(df["admissionDate"], format="%Y-%m-%d")
    df["fallDate"] = pd.to_datetime(df["fallDate"], format="%Y-%m-%d")
    columns_to_replace_nan_values = ["fallActivity", "injury"]
    df[columns_to_replace_nan_values] = df[columns_to_replace_nan_values].fillna(
        "nessuna informazione"
    )
    hospital_stays = pd.read_csv(
        path.join(CSV, "HospitalStay.csv"), sep=",", encoding="latin1"
    )
    keys1 = list(
        zip(
            list(pd.to_datetime(hospital_stays["AdmissionDate"], format="%Y-%m-%d")),
            list(hospital_stays["AnonymousID"]),
        )
    )
    keys2 = list(zip(list(df["admissionDate"]), list(df["anonymousId"])))
    if not set(keys2).issubset(set(keys1)):
        for timestamp, anon_id in set(keys2) - set(keys1):
            df = df.loc[(df["admissionDate"] != timestamp) & (df["anonymousId"] != anon_id)]
    return df


def _scalar(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.strftime("%Y-%m-%d")
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def to_records(df):
    """Each row -> nested logical-model instance; dotted cols 'a.b' -> {'a':{'b':..}}; drop NA."""
    records = []
    for _, row in df.iterrows():
        obj = {}
        for col, val in row.items():
            s = _scalar(val)
            if s is None or s == "":
                continue
            parts = col.split(".")
            cur = obj
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = s
        records.append(obj)
    return records


def main():
    makedirs(OUT, exist_ok=True)
    tables = {
        "patient": clean_patient,
        "hospital_stay": clean_hospital_stay,
        "knee": clean_knee,
        "drug": clean_drug,
        "fall": clean_fall,
    }
    summary = {}
    for name, fn in tables.items():
        df = fn()
        recs = to_records(df)
        with open(path.join(OUT, f"{name}.records.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        summary[name] = {"rows": len(df), "records": len(recs)}
        print(f"  {name:14s} rows={len(df):5d} records={len(recs):5d}")
    print("\nDone. Records in", OUT)
    return summary


if __name__ == "__main__":
    main()
