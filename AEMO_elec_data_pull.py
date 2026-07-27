from __future__ import annotations

import io
import ssl
import subprocess
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

RBA_G1_CSV_URL = "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv"
CPI_SERIES_ID = "GCPIAG"  # RBA G1: Consumer price index; All groups


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
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="mixed")
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
            value_dollars_real=("value_dollars_real", "sum"),
            days=("date", "nunique"),
        )
    )
    per_region["RRP"] = per_region["value_dollars"] / per_region["MWh"]
    per_region["RRP_real"] = per_region["value_dollars_real"] / per_region["MWh"]
    all_region = (
        per_region.groupby(list(by), as_index=False)
        .agg(
            MWh=("MWh", "sum"),
            value_dollars=("value_dollars", "sum"),
            value_dollars_real=("value_dollars_real", "sum"),
            days=("days", "max"),
        )
    )
    all_region["RRP"] = all_region["value_dollars"] / all_region["MWh"]
    all_region["RRP_real"] = all_region["value_dollars_real"] / all_region["MWh"]   
    all_region["REGION"] = "NEM"

    return pd.concat(
        [per_region, all_region[keys + ["MWh", "value_dollars", "RRP", "days", "value_dollars_real", "RRP_real"]]],
        ignore_index=True,
    )


PRICE_COLUMNS = ("RRP", "RRP_real")


def _volume(df: pd.DataFrame) -> pd.Series:
    # Weight prices by energy, falling back to demand if MWh is absent.
    return df["MWh"] if "MWh" in df.columns else df["TOTALDEMAND"]


def _nem_series(df: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    # Collapse the regions onto a single NEM series, one row per settlement
    # period: demand adds up, price is the volume-weighted average of the
    # regional prices.
    d = df.copy()
    d["_mwh"] = _volume(d)

    price_cols = [c for c in columns if c in PRICE_COLUMNS]
    other_cols = [c for c in columns if c not in PRICE_COLUMNS]
    for col in price_cols:
        d[f"_wtd_{col}"] = d[col] * d["_mwh"]

    spec = {k: (k, "first") for k in keys if k in d.columns}
    spec["_mwh"] = ("_mwh", "sum")
    spec.update({col: (col, "sum") for col in other_cols})
    spec.update({f"_wtd_{col}": (f"_wtd_{col}", "sum") for col in price_cols})

    nem = d.groupby("SETTLEMENTDATE", as_index=False).agg(**spec)
    for col in price_cols:
        nem[col] = nem[f"_wtd_{col}"] / nem["_mwh"]
    return nem.rename(columns={"_mwh": "MWh"})


def describe_by(
    df: pd.DataFrame,
    by: str | list[str],
    columns: list[str] = ["RRP_real", "TOTALDEMAND"],
) -> pd.DataFrame:
    # count/mean/min/5%/25%/50%/75%/95%/max/std for each column, grouped by
    # `by` (e.g. "yyyy" for yearly), for each REGION plus a "NEM" row built
    # from all regions combined. Price means are volume weighted (so they are
    # revenue / MWh); the other stats describe the unweighted distribution.
    # Returns columns named "<column>_<stat>".
    keys = [by] if isinstance(by, str) else list(by)
    price_cols = [c for c in columns if c in PRICE_COLUMNS]

    stats = {
        "count": "count",
        "mean": "mean",
        "min": "min",
        "5%": lambda s: s.quantile(0.05),
        "25%": lambda s: s.quantile(0.25),
        "50%": lambda s: s.quantile(0.50),
        "75%": lambda s: s.quantile(0.75),
        "95%": lambda s: s.quantile(0.95),
        "max": "max",
        "std": "std",
    }

    def _stats(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
        d = frame.copy()
        mwh = _volume(d)
        for col in price_cols:
            # Weight sums are per column so a missing price drops its own
            # volume from the denominator rather than biasing the average.
            d[f"_wtd_{col}"] = d[col] * mwh
            d[f"_w_{col}"] = mwh.where(d[col].notna())

        spec = {
            f"{col}_{name}": (col, func)
            for col in columns
            for name, func in stats.items()
        }
        spec.update(
            {f"_wtd_{col}": (f"_wtd_{col}", "sum") for col in price_cols}
            | {f"_w_{col}": (f"_w_{col}", "sum") for col in price_cols}
        )

        out = d.groupby(group_keys, as_index=False).agg(**spec)
        for col in price_cols:
            out[f"{col}_mean"] = out[f"_wtd_{col}"] / out[f"_w_{col}"]
        return out.drop(columns=[c for c in out.columns if c.startswith(("_wtd_", "_w_"))])

    per_region = _stats(df, ["REGION"] + keys)
    all_region = _stats(_nem_series(df, keys, columns), keys)
    all_region["REGION"] = "NEM"

    out = pd.concat([per_region, all_region[per_region.columns]], ignore_index=True)
    return out.sort_values(keys + ["REGION"]).reset_index(drop=True)


def plot_prices(df: pd.DataFrame, label_every: int = 5) -> None:
    import matplotlib.pyplot as plt

    time_col = next(
        (c for c in ("yyyy", "yyyy_qtr", "yyyymm") if c in df.columns),
        None,
    )
    if time_col is None:
        raise ValueError("df must contain one of yyyy, yyyy_qtr, yyyymm")

    def _label(v: object) -> str:
        s = str(v)
        if time_col == "yyyymm" and len(s) == 6 and s.isdigit():
            return pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1).strftime("%b %Y")
        if time_col == "yyyy_qtr" and "_" in s:
            yr, q = s.split("_", 1)
            return f"{q} {yr}"
        return s

    d = df.sort_values(time_col).reset_index(drop=True)
    positions = range(len(d))
    region = d["REGION"].iloc[0] if "REGION" in d.columns else ""

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(positions, d["RRP"], color="C0", marker="o", label="RRP (nominal)", zorder=3)
    if "RRP_real" in d.columns:
        ax1.plot(positions, d["RRP_real"], color="C3", marker="o", label="RRP (real)", zorder=3)

    tick_positions = list(positions)[::label_every]
    tick_labels = [_label(v) for v in d[time_col].iloc[::label_every]]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax1.set_xlabel(time_col)
    ax1.set_ylabel("RRP ($/MWh)")
    ax1.set_title(f"Electricity price & volume — {region} ({time_col})")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left")
    fig.tight_layout()
    plt.show()


def fetch_cpi(
    start_yyyymm: str = "200001",
    cache_path: str | Path = "data/cpi.csv",
) -> pd.DataFrame:
    # RBA G1 CSV: 9 metadata rows, then a "Series ID" header row, then
    # quarterly data with dates in DD/MM/YYYY. Australia's CPI is quarterly;
    # monthly rows here are forward-filled from the most recent quarterly print.
    print(f"fetching CPI: {RBA_G1_CSV_URL}")
    result = subprocess.run(
        ["curl.exe", "-sSL", "--max-time", "30", RBA_G1_CSV_URL],
        capture_output=True,
        check=True,
    )
    raw = result.stdout
    print(f"fetched CPI: {len(raw)} bytes")

    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(raw)
    print(f"cached CPI: {cache}")

    raw_df = pd.read_csv(io.BytesIO(raw), skiprows=10, header=0)
    raw_df = raw_df.rename(columns={raw_df.columns[0]: "date"})

    if CPI_SERIES_ID not in raw_df.columns:
        raise RuntimeError(
            f"{CPI_SERIES_ID} not in RBA G1 columns: {list(raw_df.columns)[:5]}..."
        )

    cpi_q = raw_df[["date", CPI_SERIES_ID]].rename(columns={CPI_SERIES_ID: "cpi"})
    cpi_q["date"] = pd.to_datetime(cpi_q["date"], dayfirst=True, errors="coerce")
    cpi_q["cpi"] = pd.to_numeric(cpi_q["cpi"], errors="coerce")
    cpi_q = cpi_q.dropna().sort_values("date")
    cpi_q["date"] = cpi_q["date"].dt.to_period("M").dt.to_timestamp()
    cpi_q = cpi_q.drop_duplicates("date", keep="last").set_index("date")

    start = pd.Timestamp(_parse_yyyymm(start_yyyymm))
    # Seed one quarter earlier so the first months of `start` can ffill.
    seed = cpi_q.index[cpi_q.index <= start].max()
    lower = seed if pd.notna(seed) else cpi_q.index.min()

    monthly_idx = pd.date_range(
        lower,
        pd.Timestamp.today().to_period("M").to_timestamp(),
        freq="MS",
    )
    monthly = cpi_q.reindex(monthly_idx).ffill().rename_axis("date").reset_index()
    monthly = monthly[monthly["date"] >= start].reset_index(drop=True)
    monthly["yyyymm"] = monthly["date"].dt.strftime("%Y%m")
    return monthly


def add_real_values(
    df: pd.DataFrame,
    cpi: pd.DataFrame,
    base_yyyymm: str | None = None,
) -> pd.DataFrame:
    # Real values expressed in dollars of `base_yyyymm` (defaults to the
    # latest CPI print, i.e. "in today's dollars"). Merges CPI on yyyymm.
    cpi_lookup = cpi[["yyyymm", "cpi"]].copy()
    cpi_lookup["yyyymm"] = cpi_lookup["yyyymm"].astype(str)
    base = (
        cpi_lookup.loc[cpi_lookup["yyyymm"] == base_yyyymm, "cpi"].iloc[0]
        if base_yyyymm is not None
        else cpi_lookup["cpi"].iloc[-1]
    )

    df = df.copy()
    df["yyyymm"] = df["yyyymm"].astype(str)
    df = df.merge(cpi_lookup, on="yyyymm", how="left")
    df["RRP_real"] = df["RRP"] * base / df["cpi"]
    df["value_dollars_real"] = df["value_dollars"] * base / df["cpi"]
    return df


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
    date_start = "202607"
    date_end = "202607"
    Use30min = True
    fetch_price_and_demand(date_start, date_end)
    df1 = load_price_and_demand(date_start, date_end)
    df1.drop(columns = ['PERIODTYPE'], inplace=True)
    df1 = add_period_length(df1)
    df_raw = to_30min(df1)
    path = save_price_and_demand(df_raw)
    print(f"wrote: {path}")

    df = pd.read_csv(path, parse_dates=["SETTLEMENTDATE"])

    cpi=fetch_cpi("200001")
    df = add_real_values(df, cpi)
    yearly = group_by_region(df, by=["yyyy"])
    monthly = group_by_region(df, by=["yyyymm"])
    quarterly = group_by_region(df, by=["yyyy_qtr"])
    period = group_by_region(df, by=["yyyy","time"])
    yearly_stats = describe_by(df, "yyyy")
    year_time_stats = describe_by(df, ["yyyy",'time'])
    output_dir = Path("data_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(output_dir / "yearly.csv", index=False)
    monthly.to_csv(output_dir / "monthly.csv", index=False)
    quarterly.to_csv(output_dir / "quarterly.csv", index=False)
    period.to_csv(output_dir / "period.csv", index=False)
    yearly_stats.to_csv(output_dir / "yearly_stats.csv", index=False)
    year_time_stats.to_csv(output_dir / "year_time_stats.csv", index=False)

    plot_prices(yearly[yearly.REGION == "NEM"])
    print(df.head())
    print(df.dtypes)
