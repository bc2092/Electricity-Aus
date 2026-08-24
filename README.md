# Electricity-Aus

Pulls Australian NEM electricity data and derives price and generation statistics.

## Data sources

| Source | What | Notes |
| --- | --- | --- |
| [AEMO price and demand](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data) | 30 min / 5 min regional price (RRP) and demand | Monthly CSV per region, no key needed |
| [RBA Table G1](https://www.rba.gov.au/statistics/tables/) | Consumer price index (`GCPIAG`) | Used to deflate prices to real dollars |
| [OpenElectricity](https://openelectricity.org.au/) (formerly OpenNEM) | Monthly generation energy by fuel technology | Needs an API key; see below |

## Setup

```
python -m venv .venv
.venv/Scripts/pip install pandas matplotlib
```

Get a free OpenElectricity API key from
[platform.openelectricity.org.au](https://platform.openelectricity.org.au/), then:

```
cp .env.example .env       # then paste the key into .env
```

`.env` is git ignored. The key is read from `api_key=`, then
`$OPENELECTRICITY_API_KEY`, then `.env` -- never hard code it in a tracked file.

## Running

```
.venv/Scripts/python AEMO_elec_data_pull.py
```

Adjust `date_start` / `date_end` in the `__main__` block. Outputs land in
`data_output/`: yearly, monthly, quarterly, rolling-window and distribution
statistics, plus `generation_monthly.csv`.

## Generation by fuel

```python
fetch_generation_by_fueltech("202501", "202608")                      # whole NEM
fetch_generation_by_fueltech("202501", "202608", by_region=True)      # per region
fetch_generation_by_fueltech("202501", "202608", grouping="fueltech_group")
```

`grouping="fueltech"` gives `coal_black`, `coal_brown`, `gas_ccgt`, `gas_ocgt`,
`gas_recip`, `gas_steam`, `gas_wcmg`, `hydro`, `wind`, `solar_utility`,
`solar_rooftop`, `bioenergy_biomass`, `distillate`, `pumps`, `battery`,
`battery_charging`, `battery_discharging`. `"fueltech_group"` collapses these to
`coal`, `gas`, `solar`, `wind`, `hydro` and so on.

Careful when totalling: `pumps` and `battery_charging` are **loads**, and
`battery` is the **net** of charging and discharging -- summing every row double
counts storage. Filter to the series you want first.

### Caching and the 730 day limit

Each month is cached to its own JSON under `data/openelectricity/`, so an
overlapping request reuses whatever is already on disk and only the missing
months cost an API call. The current month is never cached, since it is still
filling. Pass `overwrite=True` to refetch.

The community plan only serves **the last 730 days**. Older months come from the
public bulk endpoint, which needs no key:

```
https://data.openelectricity.org.au/v4/stats/au/NEM/energy/<year>.json
```

That file holds daily GWh per fueltech for a whole year, back to the start of the
NEM. Summed to months it agrees with the API to within about 0.2%, with a small
amount of gas reclassified between `gas_ccgt` and `gas_ocgt`. It carries no
`battery` net series -- derive it as `battery_discharging - battery_charging`.

`data/` is git ignored, so the cache does not travel with the repo. Once a month
falls outside the 730 day window your cached copy is the only one the API will
give you back -- rebuild it from the bulk endpoint if you clear the directory.
