from __future__ import annotations

import io
import json
import os
import ssl
import subprocess
from datetime import date, timedelta
from importlib.resources import path
from pathlib import Path
from urllib.parse import urlencode
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
    #print (f"fetching price and demand data for {start_yyyymm} to {end_yyyymm}...")
    #[print(x) for x in _iter_months(start, end)]

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
    # AEMO stamps every row with the END of its period, so a 00:00 stamp on the
    # 1st closes the month before. Ignore the 00:00 stamps and the first two
    # slots remaining in a month give its period length (30 minutes
    # historically, 5 minutes from Oct 2021) -- every region shares those
    # slots, hence the de-dup. That length then applies to every row belonging
    # to the month, including the 00:00 stamp on the 1st that closes it.
    ts = df["SETTLEMENTDATE"]
    is_midnight = ts.dt.time == pd.Timestamp("00:00").time()
    closes_prior_month = is_midnight & (ts.dt.day == 1)
    period_month = ts.dt.to_period("M").mask(
        closes_prior_month, ts.dt.to_period("M") - 1
    )

    def _minutes_between_first_two(times: pd.Series) -> float:
        slots = times.drop_duplicates().nsmallest(2).to_numpy()
        if len(slots) < 2:
            return float("nan")
        return (slots[1] - slots[0]) / pd.Timedelta(minutes=1)

    dated = ts[~is_midnight]
    period_len = dated.groupby(dated.dt.to_period("M")).apply(_minutes_between_first_two)

    df["period_len"] = period_month.map(period_len)
    df["period_start"] = ts - pd.to_timedelta(df["period_len"], unit="m")
    df["date"] = df["period_start"].dt.strftime("%Y-%m-%d")
    df["time"] = df["period_start"].dt.strftime("%H:%M")
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
    agg["period_start"] = agg["SETTLEMENTDATE"] - pd.Timedelta(minutes=new_len)
    agg["date"] = agg["period_start"].dt.strftime("%Y-%m-%d")
    agg["time"] = agg["period_start"].dt.strftime("%H:%M")

    return pd.concat([rest, agg], ignore_index=True, sort=False)


def  group_by_region(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
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


def rolling_by_region(
    df: pd.DataFrame,
    months: tuple[int, ...] = (1, 3, 6, 12),
    end: pd.Timestamp | str | None = None,
    years: int | None = None,
) -> pd.DataFrame:
    # Trailing 1/3/12 month averages, anchored on the most recent settlement in
    # the data (or on `end`) and then on that same date in each earlier year --
    # so an Aug 2026 file yields the 1/3/12 months to Aug 2026, to Aug 2025, to
    # Aug 2024 and so on. Anchors go back as far as the data supports unless
    # `years` caps them. Each window is (date - N months, date], so windows
    # nest within an anchor rather than tile.
    # The last date in a month is midnight on the 1st of the next month, the end of the last period.
    # Same shape as group_by_region's yearly output, with `date` (the window
    # end) in place of `yyyy` plus a `window` column holding the window length
    # in months. Aggregation is identical: volume weighted prices, nominal and
    # real, and a "NEM" row alongside the regions. A window is skipped rather
    # than reported short when it would reach back past the start of the data.
    end_ts = pd.Timestamp(end) if end is not None else df["SETTLEMENTDATE"].max()
    first_ts = df["SETTLEMENTDATE"].min()

    frames: list[pd.DataFrame] = []
    step = 0
    while years is None or step < years:
        anchor = end_ts - pd.DateOffset(years=step)
        if anchor <= first_ts:
            break
        step += 1

        for n in months:
            start_ts = anchor - pd.DateOffset(months=n)
            if start_ts < first_ts:
                continue
            window = df[
                (df["SETTLEMENTDATE"] > start_ts) & (df["SETTLEMENTDATE"] <= anchor)
            ].copy()
            if window.empty:
                continue
            # Named window_end here because df already carries a per-row `date`
            # that group_by_region counts for `days`; renamed on the way out.
            window["window_end"] = anchor
            window["window"] = n
            out = group_by_region(window, by=["window_end", "window"])
            out["window_start"] = start_ts
            frames.append(out)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).rename(columns={"window_end": "date"})
    return (
        out[
            [
                "REGION",
                "date",
                "window",
                "MWh",
                "value_dollars",
                "RRP",
                "days",
                "value_dollars_real",
                "RRP_real",
                "window_start",
            ]
        ]
        .sort_values(["window", "date", "REGION"], ascending=[True, False, True])
        .reset_index(drop=True)
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


OE_BASE_URL = "https://api.openelectricity.org.au/v4"
OE_PLAN_WINDOW_DAYS = 730
OE_API_KEY_ENV = "OPENELECTRICITY_API_KEY"
OE_ENV_FILE = Path(__file__).with_name(".env")


def _oe_api_key(api_key: str | None = None) -> str:
    # Never hard code the key here -- this file is committed. It comes from the
    # argument, the environment, or a local .env that git ignores.
    key = api_key or os.environ.get(OE_API_KEY_ENV) or _read_env_file().get(
        OE_API_KEY_ENV
    )
    if not key:
        raise RuntimeError(
            f"No OpenElectricity API key. Set ${OE_API_KEY_ENV}, put "
            f"{OE_API_KEY_ENV}=... in {OE_ENV_FILE}, or pass api_key=."
        )
    return key


def _read_env_file(path: str | Path = None) -> dict[str, str]:
    # Minimal KEY=value reader so the key can live in an untracked .env
    # without taking a dependency on python-dotenv.
    env_path = Path(path or OE_ENV_FILE)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip("\"'")
    return values


def oe_earliest_yyyymm(
    window_days: int = OE_PLAN_WINDOW_DAYS, today: date | None = None
) -> str:
    # Earliest COMPLETE month the plan will serve. The API refuses any start
    # before today - window_days; a month straddling that cutoff comes back
    # truncated rather than refused, so unless the cutoff lands exactly on the
    # 1st, the first usable month is the one after it. This moves forward each
    # day, so it is computed at run time rather than hard coded.
    cutoff = (today or date.today()) - timedelta(days=window_days)
    first_of_cutoff_month = cutoff.replace(day=1)
    if cutoff == first_of_cutoff_month:
        return f"{cutoff:%Y%m}"
    return f"{_add_months(first_of_cutoff_month, 1):%Y%m}"


def oe_clamp_start(start_yyyymm: str, **kwargs) -> str:
    # The later of the requested start and the earliest month the plan serves,
    # so a long AEMO range does not send the API a request it will reject.
    earliest = oe_earliest_yyyymm(**kwargs)
    if start_yyyymm < earliest:
        print(
            f"generation start clamped to plan window: "
            f"{start_yyyymm} -> {earliest}"
        )
        return earliest
    return start_yyyymm


def fetch_generation_by_fueltech(
    start_yyyymm: str,
    end_yyyymm: str,
    by_region: bool = False,
    grouping: str = "fueltech",
    api_key: str | None = None,
    cache_dir: str | Path = "data/openelectricity",
    overwrite: bool = False,
) -> pd.DataFrame:
    # Monthly NEM energy (MWh) by fuel technology from the OpenElectricity API
    # (the former OpenNEM). The community plan bills per call, so each month is
    # cached to its own JSON file and only the months missing from the cache
    # are fetched -- a later request for an overlapping range reuses whatever
    # is already on disk. overwrite=True refetches the whole range.
    #
    # `grouping` is "fueltech" (coal_black, gas_ccgt, wind, solar_utility,
    # solar_rooftop, battery_charging, ...) or the coarser "fueltech_group"
    # (coal, gas, wind, solar, ...). by_region=True splits by NEM region
    # instead of returning the whole-NEM total.
    #
    # Loads (battery_charging, pumps) come back as positive magnitudes, and
    # the "battery" fueltech is the net of charging and discharging (negative
    # while storage is a net load), so summing every fueltech double counts
    # storage -- filter to the series you want before totalling.
    start, end = _parse_yyyymm(start_yyyymm), _parse_yyyymm(end_yyyymm)
    if start > end:
        raise ValueError("start_yyyymm must be <= end_yyyymm")

    primary = "network_region" if by_region else "network"
    cache_dir = Path(cache_dir)
    months = [_parse_yyyymm(m) for m in _iter_months(start, end)]

    frames: list[pd.DataFrame] = []
    wanted: list[date] = []
    for month in months:
        cached = None if overwrite else _read_month_cache(
            cache_dir, primary, grouping, month
        )
        if cached is None:
            wanted.append(month)
        else:
            frames.append(cached)

    if wanted:
        # The API caps a 1M query at 732 days, so fetch the missing months in
        # runs of at most 24. A run spans from its first to its last missing
        # month, which may pull a few cached months back down with it -- still
        # cheaper than a call per month, since the cost is per call.
        for run in _runs_within_span(wanted, months_span=24):
            payload = _fetch_oe_window(
                run[0], run[-1], primary, grouping, api_key
            )
            fetched = _tidy_oe_network_data(payload, grouping)
            if fetched.empty:
                continue
            for yyyymm, rows in fetched.groupby("yyyymm", sort=True):
                _write_month_cache(cache_dir, primary, grouping, yyyymm, rows)
                if _parse_yyyymm(yyyymm) in wanted:
                    frames.append(rows)

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return (
        out.drop_duplicates(["yyyymm", "REGION", grouping])
        .sort_values(["yyyymm", "REGION", grouping])
        .reset_index(drop=True)
    )


def _runs_within_span(months: list[date], months_span: int = 24):
    # Group the missing months into runs whose first-to-last span fits the
    # API's 1M window. Counting months is not enough: two missing months
    # either side of a cached gap can still span more than the limit.
    run: list[date] = []
    for month in months:
        if run and month >= _add_months(run[0], months_span):
            yield run
            run = []
        run.append(month)
    if run:
        yield run


def _add_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def _month_cache_path(
    cache_dir: Path, primary: str, grouping: str, yyyymm: str
) -> Path:
    return cache_dir / f"NEM_energy_1M_{primary}_{grouping}_{yyyymm}.json"


def _read_month_cache(
    cache_dir: Path, primary: str, grouping: str, month: date
) -> pd.DataFrame | None:
    path = _month_cache_path(cache_dir, primary, grouping, f"{month:%Y%m}")
    if not path.exists():
        return None
    print(f"skip (exists): {path.name}")
    df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_month_cache(
    cache_dir: Path, primary: str, grouping: str, yyyymm: str, rows: pd.DataFrame
) -> None:
    # The current month is still filling, so cache only months that have
    # closed -- otherwise a partial month would be pinned on disk for good.
    if yyyymm >= date.today().strftime("%Y%m"):
        print(f"not cached (month incomplete): {yyyymm}")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _month_cache_path(cache_dir, primary, grouping, yyyymm)
    out = rows.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(out.to_json(orient="records"), encoding="utf-8")
    print(f"cached generation: {path}")


def _fetch_oe_window(
    start: date,
    end: date,
    primary: str,
    grouping: str,
    api_key: str | None,
) -> dict:
    # Buckets are stamped with their start and date_end excludes the bucket
    # sitting on it, so ask up to the 1st of the month after `end` to get
    # `end` itself and nothing further.
    params = {
        "metrics": "energy",
        "interval": "1M",
        "primary_grouping": primary,
        "secondary_grouping": grouping,
        "date_start": f"{start:%Y-%m-01}T00:00:00",
        "date_end": f"{_add_months(end, 1):%Y-%m-01}T00:00:00",
    }
    url = f"{OE_BASE_URL}/data/network/NEM?" + urlencode(params)

    print(f"fetching generation: {start:%Y%m}-{end:%Y%m} ({primary}/{grouping})")
    result = subprocess.run(
        [
            "curl.exe", "-sSL", "--max-time", "60",
            "-H", f"Authorization: Bearer {_oe_api_key(api_key)}",
            url,
        ],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    if not payload.get("success"):
        raise RuntimeError(f"OpenElectricity error: {payload}")
    return payload


def _tidy_oe_network_data(payload: dict, grouping: str) -> pd.DataFrame:
    # One row per (month, region, fueltech). Each series in the response
    # carries its grouping values in `columns` and its points as
    # [timestamp, value] pairs.
    rows: list[dict] = []
    for series in payload.get("data", []):
        unit = series.get("unit")
        for result in series.get("results", []):
            cols = result.get("columns", {})
            for stamp, value in result.get("data", []):
                rows.append(
                    {
                        "date": stamp,
                        "REGION": cols.get("region") or cols.get("network_region") or "NEM",
                        grouping: cols.get(grouping),
                        "MWh": value,
                        "unit": unit,
                    }
                )

    df = pd.DataFrame(rows, columns=["date", "REGION", grouping, "MWh", "unit"])
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(
        "Australia/Brisbane"
    ).dt.tz_localize(None)
    df["yyyymm"] = df["date"].dt.strftime("%Y%m")
    df["yyyy"] = df["date"].dt.strftime("%Y")
    df["yyyy_qtr"] = df["date"].dt.to_period("Q").astype(str).str.replace("Q", "_Q")
    df["MWh"] = pd.to_numeric(df["MWh"], errors="coerce")
    return df.sort_values(["yyyymm", "REGION", grouping]).reset_index(drop=True)


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
        #else cpi_lookup["cpi"].iloc[-1]
        else cpi_lookup.loc[cpi_lookup.yyyymm == cpi_lookup.yyyymm.max(), "cpi"].iloc[0]
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
        existing = pd.read_csv(target, parse_dates=["SETTLEMENTDATE", "period_start"])
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
    output_dir = Path("data_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_start = "202607"
    date_end = "202608"
    # OpenElectricity generation by fuel. The community plan only serves the
    # last 730 days, so start at whichever is later: the range asked for above,
    # or the earliest complete month the plan still covers.
    gen_start = oe_clamp_start(date_start)
    grouping_p = "fueltech_group"
    generation = fetch_generation_by_fueltech(gen_start, date_end, grouping = grouping_p)
    generation.to_csv(output_dir / f"generation_monthly_{grouping_p}.csv", index=False)
    grouping_p = "fueltech"
    generation = fetch_generation_by_fueltech(gen_start, date_end, grouping = grouping_p)
    generation.to_csv(output_dir / f"generation_monthly_{grouping_p}.csv", index=False)
    Use30min = True
    fetch_price_and_demand(date_start, date_end, overwrite=True)
    df1 = load_price_and_demand(date_start, date_end)
    df1.drop(columns = ['PERIODTYPE'], inplace=True)
    df1 = add_period_length(df1)
    df_raw = to_30min(df1)
    path = save_price_and_demand(df_raw)
    print(f"wrote: {path}")

    df = pd.read_csv(path, parse_dates=["SETTLEMENTDATE", "period_start"])

    cpi=fetch_cpi("200001")
    df = add_real_values(df, cpi)
    yearly = group_by_region(df, by=["yyyy"])
    monthly = group_by_region(df, by=["yyyymm"])
    quarterly = group_by_region(df, by=["yyyy_qtr"])
    period = group_by_region(df, by=["yyyy","time"])
    # End the windows at the last full month rather than part way through the
    # current one. AEMO stamps a period with its END, so the 00:00 stamp on the
    # 1st closes the month before -- the start of the latest month present is
    # therefore midnight at the end of the last full month, and anchoring there
    # keeps that month's final day in the window.
    last_month_end = df["SETTLEMENTDATE"].max().to_period("M").to_timestamp()
    rolling = rolling_by_region(df, end=last_month_end)
    print(f"rolling windows end: {(last_month_end - pd.Timedelta(days=1)).date()} 24:00")
    yearly_stats = describe_by(df, "yyyy")
    year_time_stats = describe_by(df, ["yyyy",'time'])

    yearly.to_csv(output_dir / "yearly.csv", index=False)
    monthly.to_csv(output_dir / "monthly.csv", index=False)
    quarterly.to_csv(output_dir / "quarterly.csv", index=False)
    period.to_csv(output_dir / "period.csv", index=False)
    rolling.to_csv(output_dir / "rolling.csv", index=False)
    yearly_stats.to_csv(output_dir / "yearly_stats.csv", index=False)
    
    year_time_stats.to_csv(output_dir / "year_time_stats.csv", index=False)

    #plot_prices(yearly[yearly.REGION == "NEM"])
    print(df.head())
    print(df.dtypes)
