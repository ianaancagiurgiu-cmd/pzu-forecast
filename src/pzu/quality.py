"""Raport de calitate a datelor: goluri, duplicate, zile cu numar anormal de intervale."""
import pandas as pd

from pzu.timegrid import day_index, to_local


def coverage_report(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Pe zi locala: intervale prezente vs asteptate (92/96/100 la schimbarea orei)."""
    rows = []
    for day in pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D"):
        idx = day_index(day)
        present = df.reindex(idx).notna().sum() if not df.empty else pd.Series(0, index=df.columns)
        rows.append({"day": day.date(), "expected": len(idx), **present.to_dict()})
    return pd.DataFrame(rows).set_index("day")


def summary(df: pd.DataFrame, start, end) -> dict:
    rep = coverage_report(df, start, end)
    cols = [c for c in rep.columns if c != "expected"]
    return {
        "days": len(rep),
        "duplicates": int(df.index.duplicated().sum()),
        "dst_days": rep.index[rep["expected"] != 96].tolist(),
        "missing_share": {c: float(1 - rep[c].sum() / rep["expected"].sum()) for c in cols},
        "days_incomplete": {c: int((rep[c] < rep["expected"]).sum()) for c in cols},
    }
