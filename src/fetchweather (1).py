# src/fetchweather.py
from __future__ import annotations
import os, time, math, requests
from typing import Iterable, Optional, Dict, List, Tuple
import pandas as pd
import numpy as np

# ----------------- Config / Tokens -----------------
USDA_API_KEY = os.getenv("USDA_API_KEY", "").strip()
NOAA_TOKEN   = os.getenv("NOAA_TOKEN", "").strip()

USDA_BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
NOAA_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

# 7 major soybean states (WAOB focus)
SEVEN_STATES = ("IA","IL","IN","OH","MO","MN","NE")

# FIPS map for NOAA locationid=FIPS:XX
FIPS_2 = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","FL":"12",
    "GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21","LA":"22",
    "ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30","NE":"31",
    "NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39","OK":"40",
    "OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49","VT":"50",
    "VA":"51","WA":"53","WV":"54","WI":"55","WY":"56"
}

# NOAA NCLIMDIV datatypes
NOAA_DATATYPES = {"TAVG":"TAVG", "TPCP":"TPCP"}  # Monthly temperature average, total precip

# ----------------- Utilities -----------------
def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state"] = out["state"].astype(str).str.strip().upper()
    out["year"]  = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def _usda_get(params: Dict, page_size: int = 50000) -> pd.DataFrame:
    if not USDA_API_KEY:
        raise RuntimeError("USDA_API_KEY not set in environment")
    q = dict(params, key=USDA_API_KEY, page_size=page_size)
    out, page = [], 1
    while True:
        qp = dict(q, page=page)
        r = requests.get(USDA_BASE, params=qp, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"USDA error {r.status_code}: {r.text[:300]}")
        data = r.json().get("data", [])
        if not data: break
        out.extend(data)
        if len(data) < page_size: break
        page += 1
    return pd.DataFrame(out)

def _noaa_get(datasetid: str, datatypeid: str, locationid: str, start: str, end: str, 
              limit: int = 1000) -> pd.DataFrame:
    if not NOAA_TOKEN:
        raise RuntimeError("NOAA_TOKEN not set in environment")
    headers = {"token": NOAA_TOKEN}
    out, offset = [], 1
    while True:
        params = {
            "datasetid": datasetid,
            "datatypeid": datatypeid,
            "locationid": locationid,
            "startdate": start,
            "enddate": end,
            "limit": limit,
            "offset": offset,
            "units": "standard"  # Fahrenheit / inches for NCLIMDIV
        }
        r = requests.get(NOAA_BASE, headers=headers, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(1.0);  # rate limit backoff
            continue
        if r.status_code != 200:
            raise RuntimeError(f"NOAA error {r.status_code}: {r.text[:300]}")
        js = r.json()
        results = js.get("results", [])
        if not results:
            break
        out.extend(results)
        if len(results) < limit:
            break
        offset += limit
    return pd.DataFrame(out)

# ----------------- USDA: Yield & Harvested area -----------------
def fetch_yield_and_area(year_from: int, year_to: int, states: Iterable[str]) -> pd.DataFrame:
    params = {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "SOYBEANS",
        "agg_level_desc": "STATE",
        "state_alpha": ",".join(states),
        "year__GE": int(year_from),
        "year__LE": int(year_to),
    }
    # Yield (bu/acre)
    y_params = dict(params,
                    statisticcat_desc="YIELD",
                    unit_desc="BU / ACRE")
    df_y = _usda_get(y_params)
    df_y = df_y.rename(columns={"Value":"yield_bu_acre","state_alpha":"state"})
    # numeric conversion
    df_y["yield_bu_acre"] = pd.to_numeric(df_y["yield_bu_acre"].astype(str).str.replace(",","",regex=False), errors="coerce")
    df_y = df_y[["state","year","yield_bu_acre"]]

    # Harvested acres (convert to hectares as well)
    a_params = dict(params,
                    statisticcat_desc="AREA HARVESTED",
                    unit_desc="ACRES")
    df_a = _usda_get(a_params)
    df_a = df_a.rename(columns={"Value":"acres_harvested","state_alpha":"state"})
    df_a["acres_harvested"] = pd.to_numeric(df_a["acres_harvested"].astype(str).str.replace(",","",regex=False), errors="coerce")
    df_a["harvest_ha"] = df_a["acres_harvested"] * 0.40468564224
    df_a = df_a[["state","year","acres_harvested","harvest_ha"]]

    out = (df_y.merge(df_a, on=["state","year"], how="outer"))
    return _norm_keys(out)

# ----------------- NOAA: Monthly weather (state-level) -----------------
def fetch_noaa_monthly(year_from: int, year_to: int, states: Iterable[str]) -> pd.DataFrame:
    rows = []
    for st in states:
        fips = FIPS_2.get(st)
        if not fips:
            continue
        loc = f"FIPS:{fips}"
        # Temperature average
        df_t = _noaa_get("NCLIMDIV", "TAVG", loc, f"{year_from}-01-01", f"{year_to}-12-31")
        # Total precipitation
        df_p = _noaa_get("NCLIMDIV", "TPCP", loc, f"{year_from}-01-01", f"{year_to}-12-31")
        if df_t.empty or df_p.empty:
            continue
        df_t["state"] = st; df_p["state"] = st
        # Parse date and year, month
        for d in (df_t, df_p):
            d["date"] = pd.to_datetime(d["date"])
            d["year"] = d["date"].dt.year
            d["month"] = d["date"].dt.month
        # Merge monthly temp/prec
        m = df_t.merge(df_p, on=["state","date","year","month","station"], how="outer", suffixes=("_t","_p"))
        # NOAA returns tenths for some datasets; for NCLIMDIV units='standard' usually returns in °F and inches.
        m = m.rename(columns={"value_t":"tavg_f","value_p":"pcp_in"})
        rows.append(m[["state","year","month","tavg_f","pcp_in"]])
    if not rows:
        return pd.DataFrame(columns=["state","year","month","tavg_f","pcp_in"])
    df = pd.concat(rows, ignore_index=True)
    return _norm_keys(df)

# ----------------- Build WAOB-like features from monthly -----------------
def build_waob_from_monthlies(monthly: pd.DataFrame, df_yield_area: pd.DataFrame,
                              base_trend_year: int = 1987) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    m = monthly.copy()
    # July–August means
    ja = m[m["month"].isin([7,8])].groupby(["state","year"], as_index=False).agg(
        temp_JA=("tavg_f","mean"),
        prec_JA=("pcp_in","mean")
    )
    ja["prec_JA_sq"] = ja["prec_JA"]**2

    # June precipitation shortfall: positive if June below its p10 over history (per state)
    jun = (m[m["month"].eq(6)]
           .groupby(["state","year"], as_index=False)
           .agg(prec_Jun=("pcp_in","mean")))
    # Compute p10 per state on training span (use all years available)
    p10 = jun.groupby("state")["prec_Jun"].quantile(0.10).rename("jun_p10").reset_index()
    jun = jun.merge(p10, on="state", how="left")
    jun["jun_shortfall"] = (jun["jun_p10"] - jun["prec_Jun"]).clip(lower=0.0)

    out = (ja.merge(jun[["state","year","jun_shortfall"]], on=["state","year"], how="left")
             .merge(df_yield_area, on=["state","year"], how="left"))
    # trend and dummy
    out["trend"] = out["year"].astype(int) - int(base_trend_year)
    out["dummy_2003"] = (out["year"].astype(int) == 2003).astype(float)
    # final order
    cols = ["state","year","yield_bu_acre","harvest_ha","acres_harvested",
            "trend","jun_shortfall","temp_JA","prec_JA","prec_JA_sq","dummy_2003"]
    return out[cols].sort_values(["state","year"]).reset_index(drop=True)

# ----------------- Add US weighted line -----------------
def add_us_weighted_row(df_in: pd.DataFrame) -> pd.DataFrame:
    if "harvest_ha" not in df_in.columns:
        raise ValueError("harvest_ha is required to compute US weights.")
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["harvest_ha","year"]]
    def wmean(g: pd.DataFrame) -> pd.Series:
        w = g["harvest_ha"].fillna(0.0).astype(float)
        out = {c: (np.average(g[c].astype(float), weights=w) if w.sum() > 0 else np.nan)
               for c in num_cols}
        out["harvest_ha"] = float(w.sum())
        out["acres_harvested"] = float(g.get("acres_harvested", pd.Series([0]*len(g))).fillna(0).sum())
        return pd.Series(out)
    us = df_in.groupby("year", as_index=True).apply(wmean).reset_index()
    us["state"] = "US"
    return pd.concat([df_in, us], ignore_index=True, sort=False)
