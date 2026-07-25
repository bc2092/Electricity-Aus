from __future__ import annotations

from datetime import date
from importlib.resources import path
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

REGIONS = ("NSW1", "QLD1", "VIC1", "TAS1", "SA1")
BASE_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _parse_yyyymm(value: str) -> date:
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"Expected yyyymm, got {value!r}")
    year, month = int(value[:4]), int(value[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in {value!r}")
    return date(year, month, 1)


def _iter_months(start: date, end: date):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            month = 1
            year += 1


def fetch_price_and_demand(
    start_yyyymm: str,
    end_yyyymm: str,
    regions: tuple[str, ...] = REGIONS,
    output_dir: str | Path = "monthly_files",
    overwrite: bool = False,
) -> list[Path]:
    start, end = _parse_yyyymm(start_yyyymm), _parse_yyyymm(end_yyyymm)
    if start > end:
        raise ValueError("start_yyyymm must be <= end_yyyymm")

    out_dir = Path(output_dir)
    #out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for yyyymm in _iter_months(start, end):
        for region in regions:
            filename = f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
            target = out_dir / filename
            if target.exists() and not overwrite:
                print(f"skip (exists): {filename}")
                saved.append(target)
                continue

            url = f"{BASE_URL}/{filename}"
            request = Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urlopen(request) as response:
                    target.write_bytes(response.read())
                print(f"downloaded: {filename}")
                saved.append(target)
            except HTTPError as e:
                print(f"failed ({e.code}): {filename}")
            except URLError as e:
                print(f"failed ({e.reason}): {filename}")

    return saved


def load_price_and_demand(
    start_yyyymm: str,
    end_yyyymm: str,
    regions: tuple[str, ...] = REGIONS,
    input_dir: str | Path = "monthly_files",
) -> pd.DataFrame:
    start, end = _parse_yyyymm(start_yyyymm), _parse_yyyymm(end_yyyymm)
    if start > end:
        raise ValueError("start_yyyymm must be <= end_yyyymm")

    in_dir = Path(input_dir)
    frames: list[pd.DataFrame] = []
    for yyyymm in _iter_months(start, end):
        for region in regions:
            path = in_dir / f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
            if not path.exists():
                print(f"missing: {path.name}")
                continue
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "SETTLEMENTDATE" in df.columns:
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    return df


def add_period_length(df: pd.DataFrame) -> pd.DataFrame:
    # AEMO stamps each period with its end time, so 00:00 on the 1st of a
    # month is actually the final period of the prior month. Shift those
    # rows back by one nanosecond so they group with the correct month.
    ts = df["SETTLEMENTDATE"]
    is_month_boundary = (ts.dt.day == 1) & (ts.dt.time == pd.Timestamp("00:00").time())
    effective_month = ts.where(~is_month_boundary, ts - pd.Timedelta(nanoseconds=1)).dt.to_period("M")

    def _minutes_between_first_two(times: pd.Series) -> float:
        unique_sorted = times.drop_duplicates().sort_values().to_numpy()
        if len(unique_sorted) < 2:
            return float("nan")
        return (unique_sorted[1] - unique_sorted[0]) / pd.Timedelta(minutes=1)

    df["period_len"] = effective_month.map(
        df.groupby(effective_month)["SETTLEMENTDATE"].apply(_minutes_between_first_two)
    )

    period_start = df["SETTLEMENTDATE"] - pd.to_timedelta(df["period_len"], unit="m")
    df["date"] = period_start.dt.strftime("%Y-%m-%d")
    df["time"] = period_start.dt.strftime("%H:%M")
    df['yyyy'] = df["date"].str.replace("-", "").str.slice(0, 4)
    df['yyyymm'] = df["date"].str.replace("-", "").str.slice(0, 6)
    df['yyyy_qtr'] = df["date"].str.replace("-", "").str.slice(0, 6).apply(lambda x: f"{x[:4]}_Q{((int(x[4:6])-1)//3)+1}")
    df["MWh"] = df["period_len"] * df["TOTALDEMAND"] / 60
    df["value_dollars"] = df["RRP"] * df["MWh"]
    return df


def to_30min(df: pd.DataFrame) -> pd.DataFrame:
    five_mask = df["period_len"] == 5
    new_len = 30
    if not five_mask.any():
        return df

    five = df.loc[five_mask].copy()
    rest = df.loc[~five_mask]

    bucket = five["SETTLEMENTDATE"].dt.ceil(str(new_len)+"min")
    agg = (
        five.groupby([five["REGION"], bucket])
        .agg(MWh=("MWh", "sum"), value_dollars=("value_dollars", "sum"), yyyy=("yyyy", "first"), yyyymm=("yyyymm", "first"), yyyy_qtr=("yyyy_qtr", "first"))
        .reset_index()
    )
    agg["RRP"] = agg["value_dollars"] / agg["MWh"]
    agg["TOTALDEMAND"] = agg["MWh"] * 60 / new_len
    agg["period_len"] = new_len
    period_start = agg["SETTLEMENTDATE"] - pd.Timedelta(minutes=new_len)
    agg["date"] = period_start.dt.strftime("%Y-%m-%d")
    agg["time"] = period_start.dt.strftime("%H:%M")

    return pd.concat([rest, agg], ignore_index=True, sort=False)


def group_by_region(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    keys = ["REGION"] + list(by)
    per_region = (
        df.groupby(keys, as_index=False)
        .agg(
            MWh=("MWh", "sum"),
            value_dollars=("value_dollars", "sum"),
            days=("date", "nunique"),
        )
    )
    per_region["RRP"] = per_region["value_dollars"] / per_region["MWh"]

    all_region = (
        per_region.groupby(list(by), as_index=False)
        .agg(
            MWh=("MWh", "sum"),
            value_dollars=("value_dollars", "sum"),
            days=("days", "max"),
        )
    )
    all_region["RRP"] = all_region["value_dollars"] / all_region["MWh"]
    all_region["REGION"] = "NEM"

    return pd.concat(
        [per_region, all_region[keys + ["MWh", "value_dollars", "RRP", "days"]]],
        ignore_index=True,
    )


def save_price_and_demand(
    df: pd.DataFrame,
    output_dir: str | Path = "data",
    filename: str = "AEMO_30min.csv",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename

    if target.exists():
        existing = pd.read_csv(target, parse_dates=["SETTLEMENTDATE"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["REGION", "SETTLEMENTDATE"], keep="last"
        )
    else:
        combined = df.copy()

    combined_out = combined.sort_values(["yyyymm", "REGION","SETTLEMENTDATE"]).reset_index(drop=True)
    combined_out.to_csv(target, index=False)
    return target


if __name__ == "__main__":
    date_start = "202001"
    date_end = "202412"
    Use30min = True
    fetch_price_and_demand(date_start, date_end)
    df1 = load_price_and_demand(date_start, date_end)
    df1.drop(columns = ['PERIODTYPE'], inplace=True)
    df1 = add_period_length(df1)
    df_raw = to_30min(df1)
    path = save_price_and_demand(df_raw)
    print(f"wrote: {path}")

    df = pd.read_csv(path, parse_dates=["SETTLEMENTDATE"])

    yearly = group_by_region(df, by=["yyyy"])
    monthly = group_by_region(df, by=["yyyymm"])
    quarterly = group_by_region(df, by=["yyyy_qtr"])

    print(df.head())
    print(df.dtypes)
