import os
import pandas as pd
import numpy as np
import requests

CLIMDIV_PRCP_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpnst-v1.0.0-20250905"
CLIMDIV_TEMP_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-20250905"

USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

DEFAULT_STATES = ["IA","IL","IN","OH","MO","MN","NE"]
USDA_API_KEY = "D1ABF2AD-362D-346E-A641-93A2FA6ED6D8"

STATE_CODES = {
        1:"AL",  2:"AZ",  3:"AR",  4:"CA",  5:"CO",  6:"CT",  7:"DE",  8:"FL",  9:"GA",
    10:"ID", 11:"IL", 12:"IN", 13:"IA", 14:"KS", 15:"KY", 16:"LA", 17:"ME", 18:"MD",
    19:"MA", 20:"MI", 21:"MN", 22:"MS", 23:"MO", 24:"MT", 25:"NE", 26:"NV", 27:"NH",
    28:"NJ", 29:"NM", 30:"NY", 31:"NC", 32:"ND", 33:"OH", 34:"OK", 35:"OR", 36:"PA",
    37:"RI", 38:"SC", 39:"SD", 40:"TN", 41:"TX", 42:"UT", 43:"VT", 44:"VA", 45:"WA",
    46:"WV", 47:"WI", 48:"WY", 49:"HI", 50:"AK", 110:"US"
    }

ACRE_TO_HA = 0.40468564224
BUAC_TO_THA = 0.0272155422 / 0.40468564224

START_YEAR = 1988
END_YEAR = 2024
BASE_YEAR = START_YEAR - 1
WAOB_STATES = ["IA","IL","IN","OH","MO","MN","NE"]

def ensure_dirs():
    for p in ["data/raw","data/interim","data/processed", "data/results"]:
        os.makedirs(p, exist_ok=True)

ensure_dirs()

def _parse_climdiv_lines(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            state_code = int(line[0:3])
            year       = int(line[6:10])
        except ValueError:
            continue

        if state_code > 50 and state_code != 110:
            continue

        parts = line[10:].split()
        if len(parts) < 12:
            continue

        try:
            vals = [float(x) for x in parts[:12]]
        except ValueError:
            continue

        for m, v in enumerate(vals, start=1):
            rows.append((state_code, year, m, v))

    return pd.DataFrame(rows, columns=["state_code", "year", "month", "value"])


def _climdiv_wide(url: str, start: int, end: int, states: list[str], prefix: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = _parse_climdiv_lines(resp.text)

    df["state"] = df["state_code"].map(STATE_CODES)
    df = df.dropna(subset=["state"])
    df = df[df["year"].between(start, end) & df["state"].isin(states)].copy()

    wide = df.pivot_table(index=["state", "year"], columns="month", values="value", aggfunc="mean").reset_index()

    month_names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    wide.columns = ["state", "year"] + [f"{prefix}_{name}" for name in month_names]
    wide.to_csv(f"data/raw/{prefix}_df.csv")

    return wide


def build_weather_table(start: int, end: int, states: list[str]) -> pd.DataFrame:
    T = _climdiv_wide(CLIMDIV_TEMP_URL, start, end, states, prefix="t")
    P = _climdiv_wide(CLIMDIV_PRCP_URL, start, end, states, prefix="p")

    W = pd.merge(T, P, on=["state", "year"], how="inner")

    W["temp_JA"]   = W[["t_jul", "t_aug"]].mean(axis=1)
    W["prec_jun"]  = W["p_jun"]
    W["prec_JA"] = W[["p_jul","p_aug"]].mean(axis=1)
    W["prec_JA_sq"] = W["prec_JA"] ** 2

    return W[["state", "year", "temp_JA", "prec_JA", "prec_JA_sq", "prec_jun"]]

def usda_quickstats(params):
    key = USDA_API_KEY
    if not key:
        raise RuntimeError("USDA API key not found")
    params = dict(params)
    params["key"] = key
    r = requests.get(USDA_BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])


def _to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

def get_soy_state_metric(start, end, states, metric, unit, out_col, agg="sum"):
    params = {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": metric,   
        "unit_desc": unit,             
        "agg_level_desc": "STATE",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "source_desc": "SURVEY",
        "year__GE": start,
        "year__LE": end
    }
    df = usda_quickstats(params)
    df = df[df["state_alpha"].isin(states)].copy()
    df["Value"] = _to_numeric(df["Value"])

    if agg == "mean":
        out = df.groupby(["state_alpha", "year"], as_index=False)["Value"].mean()
    else:
        out = df.groupby(["state_alpha", "year"], as_index=False)["Value"].sum()

    out.rename(columns={"state_alpha": "state", "Value": out_col}, inplace=True)
    return out

def get_soy_national_yield(start, end, unit="BU / ACRE"):
    params = {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "YIELD",
        "unit_desc": unit,             # "BU / ACRE" (source officielle)
        "agg_level_desc": "NATIONAL",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "source_desc": "SURVEY",
        "year__GE": start,
        "year__LE": end
    }
    df = usda_quickstats(params).copy()

    df["Value"] = _to_numeric(df["Value"])
    out = (df.groupby("year", as_index=False)["Value"]
             .mean() 
             .rename(columns={"Value": "yield_bu_acre"}))
    out["yield_t_ha"] = out["yield_bu_acre"] * BUAC_TO_THA
    return out[["year", "yield_bu_acre", "yield_t_ha"]]

def add_shortfall(df):
    stats = (df.groupby('state')['prec_jun']
               .agg(mean_jun='mean', p10=lambda x: np.percentile(x, 10))
               .reset_index())
    df = df.merge(stats, on='state', how='left')
    df['jun_shortfall'] = np.where(df['prec_jun'] <= df['p10'],df['mean_jun'] - df['prec_jun'],0.0)
    return df.drop(columns=['mean_jun','p10'])

def build_state_features(start, end, states):
    W = build_weather_table(start, end, states) 

    Y = get_soy_state_metric(start, end, states,
                             "YIELD", "BU / ACRE", "yield_bu_acre", "mean")
    Y.to_csv("data/raw/yield_state_bu_ac.csv", index=False)

    A = get_soy_state_metric(start, end, states,
                             "AREA HARVESTED", "ACRES", "acres_harvested", "sum")
    A["harvest_ha"] = A["acres_harvested"] * ACRE_TO_HA
    A.to_csv("data/raw/harvest_state_acres_ha.csv", index=False)

    df = W.merge(Y, on=["state","year"], how="left").merge(A, on=["state","year"], how="left")

    df = add_shortfall(df)

    df["trend"] = df["year"] - BASE_YEAR
    df["dummy_2003"] = (df["year"] == 2003).astype(int)

    cols = [
        "state","year",
        "yield_bu_acre",  
        "trend","jun_shortfall","temp_JA","prec_JA","prec_JA_sq",
        "dummy_2003",
        "acres_harvested","harvest_ha"
    ]

    print("State features Build")
    return df[cols]

def aggregate_national(df_state, df_us_yield, method="weighted", weight_col="harvest_ha"):
    rows = []
    for year, grp in df_state.groupby("year"):
        if (method == "weighted") and (weight_col in grp.columns):
            w = grp[weight_col].astype(float)
        else:
            w = pd.Series(1.0, index=grp.index)

        def wavg(s): return np.average(s.astype(float), weights=w)
        p_ja = wavg(grp["prec_JA"])

        rows.append({
            "year": int(year),
            "trend": int(year - BASE_YEAR),
            "jun_shortfall": wavg(grp["jun_shortfall"]),
            "temp_JA": wavg(grp["temp_JA"]),
            "prec_JA": p_ja,
            "prec_JA_sq": p_ja ** 2,             
            "dummy_2003": int(year == 2003),
            "harvest_total_ha": float(w.sum()),
            "harvest_total_acres": float(grp["acres_harvested"].sum())
        })

    X = pd.DataFrame(rows).sort_values("year")
    X = X.merge(df_us_yield, on="year", how="left")

    order = ["year","yield_bu_acre","yield_t_ha","trend",
            "jun_shortfall","temp_JA","prec_JA","prec_JA_sq",
            "dummy_2003","harvest_total_acres","harvest_total_ha"]
    return X[order]

def add_us_weighted_row(df: pd.DataFrame, weight_col: str = "harvest_ha") -> pd.DataFrame:
    if weight_col not in df.columns:
        raise ValueError(f"Colonne '{weight_col}' introuvable dans df")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [weight_col, "year"]]

    def weighted_mean(g: pd.DataFrame) -> pd.Series:
        w = g[weight_col].fillna(0).astype(float)
        out = {c: (np.average(g[c].astype(float), weights=w) if w.sum() > 0 else np.nan)
               for c in num_cols}
        out[weight_col] = float(w.sum()) 
        return pd.Series(out)

    df_us = (df.groupby("year", as_index=True).apply(weighted_mean).reset_index())
    df_us["state"] = "US"

    df_final = pd.concat([df, df_us], ignore_index=True, sort=False)
    return df_final


df_states = build_state_features(START_YEAR, END_YEAR, WAOB_STATES)
df_states = add_us_weighted_row(df_states, weight_col="harvest_ha")
df_states.to_csv("data/processed/waob_features_states.csv", index=False)

print(df_states.tail())