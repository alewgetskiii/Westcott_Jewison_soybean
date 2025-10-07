#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import requests
import argparse

CLIMDIV_BASE = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv"
CLIMDIV_TEMP_URL = f"{CLIMDIV_BASE}/climdiv-norm-tmpcst-v1.0.0-20250905"
CLIMDIV_PRCP_URL = f"{CLIMDIV_BASE}/climdiv-pcpnst-v1.0.0-20250905"
USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

DEFAULT_STATES = ["IA","IL","IN","OH","MO","MN","NE"]
USDA_KEY = "D1ABF2AD-362D-346E-A641-93A2FA6ED6D8"


def ensure_dirs():
    for p in ["data/raw","data/interim","data/processed"]:
        os.makedirs(p, exist_ok=True)



def month_name(m):
    return ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1988)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--states", type=str, default=",".join(DEFAULT_STATES))
    ap.add_argument("--base_year", type=int, default=1987)
    ap.add_argument("--national_method", type=str, choices=["weighted","simple"], default="weighted")
    return ap.parse_args()


def load_climdiv_txt(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    print(r)
    lines = r.text.strip().splitlines()
    rows = []
    for line in lines:
        p = line.split()
        if len(p) >= 15:
            rows.append([p[0], p[1], int(p[2])] + [float(x) for x in p[3:15]])
    cols = ["state_code","division","year"] + [f"m{m:02d}" for m in range(1,13)]
    print("ok load_climdiv_txt")
    df = pd.DataFrame(rows, columns=cols)
    print(df.head())
    return df


def climdiv_statewide_month(df):
    grp = df.groupby(["state_code","year"]).mean(numeric_only=True).reset_index()
    print(grp.head())
    long = grp.melt(id_vars=["state_code","year"], value_vars=[f"m{m:02d}" for m in range(1,13)],
                    var_name="month", value_name="value")
    long["month"] = long["month"].str[1:].astype(int)
    print("ok climdiv srtatewide mothn")
    return long


def map_state_code_to_abbr(code):
    mapping = {
        "01":"AL","02":"AZ","03":"AR","04":"CA","05":"CO","06":"CT","07":"DE","08":"FL","09":"GA",
        "10":"ID","11":"IL","12":"IN","13":"IA","14":"KS","15":"KY","16":"LA","17":"ME","18":"MD",
        "19":"MA","20":"MI","21":"MN","22":"MS","23":"MO","24":"MT","25":"NE","26":"NV","27":"NH",
        "28":"NJ","29":"NM","30":"NY","31":"NC","32":"ND","33":"OH","34":"OK","35":"OR","36":"PA",
        "37":"RI","38":"SC","39":"SD","40":"TN","41":"TX","42":"UT","43":"VT","44":"VA","45":"WA",
        "46":"WV","47":"WI","48":"WY"
    }
    print("ok map state")
    return mapping.get(code, None)


def build_weather_table(start, end, states):
    tdf = load_climdiv_txt(CLIMDIV_TEMP_URL)
    pdf = load_climdiv_txt(CLIMDIV_PRCP_URL)
    t_long = climdiv_statewide_month(tdf)
    p_long = climdiv_statewide_month(pdf)
    t_long["state"] = t_long["state_code"].apply(map_state_code_to_abbr)
    p_long["state"] = p_long["state_code"].apply(map_state_code_to_abbr)
    t_long = t_long.dropna(subset=["state"])
    p_long = p_long.dropna(subset=["state"])
    t_long = t_long[(t_long["year"].between(start, end)) & (t_long["state"].isin(states))]
    p_long = p_long[(p_long["year"].between(start, end)) & (p_long["state"].isin(states))]
    T = t_long.pivot_table(index=["state","year"], columns="month", values="value").reset_index()
    P = p_long.pivot_table(index=["state","year"], columns="month", values="value").reset_index()
    T.columns = ["state","year"] + [f"t_{month_name(m).lower()}" for m in range(1,13)]
    P.columns = ["state","year"] + [f"p_{month_name(m).lower()}" for m in range(1,13)]
    W = pd.merge(T, P, on=["state","year"], how="inner")
    W["temp_JA"] = W[["t_jul","t_aug"]].mean(axis=1)
    W["prec_JA"] = W[["p_jul","p_aug"]].mean(axis=1)
    W["prec_JA_sq"] = W["prec_JA"]**2
    W["prec_jun"] = W["p_jun"]
    print("ok build weather")
    return W[["state","year","temp_JA","prec_JA","prec_JA_sq","prec_jun"]]


def usda_quickstats(params):
    key = os.environ.get(USDA_KEY_ENV, "")
    if not key:
        raise RuntimeError("Set USDA_QUICKSTATS_KEY env var")
    params = dict(params)
    params["key"] = key
    r = requests.get(USDA_BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json()["data"])


def get_soy_yield_state(start, end, states):
    params = {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "STATE",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "source_desc": "SURVEY",
        "year__GE": start,
        "year__LE": end
    }
    df = usda_quickstats(params)
    df = df[df["state_alpha"].isin(states)].copy()
    df["Value"] = pd.to_numeric(df["Value"].str.replace(",",""), errors="coerce")
    out = df.groupby(["state_alpha","year"]).agg(yield_bu_acre=("Value","mean")).reset_index()
    out.rename(columns={"state_alpha":"state"}, inplace=True)
    print("ok get soy yeils")
    return out


def get_soy_acres_harvested_state(start, end, states):
    params = {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "AREA HARVESTED",
        "unit_desc": "ACRES",
        "agg_level_desc": "STATE",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "source_desc": "SURVEY",
        "year__GE": start,
        "year__LE": end
    }
    df = usda_quickstats(params)
    df = df[df["state_alpha"].isin(states)].copy()
    df["Value"] = pd.to_numeric(df["Value"].str.replace(",",""), errors="coerce")
    out = df.groupby(["state_alpha","year"]).agg(acres_harvested=("Value","sum")).reset_index()
    out.rename(columns={"state_alpha":"state"}, inplace=True)
    print("ok get soy acree")
    return out


def add_shortfall(df):
    stats = df.groupby("state")["prec_jun"].agg(mean_jun="mean", p10=lambda x: np.percentile(x, 10)).reset_index()
    df = df.merge(stats, on="state", how="left")
    df["jun_shortfall"] = np.where(df["prec_jun"] <= df["p10"], df["prec_jun"] - df["mean_jun"], 0.0)
    print("ok add shortfall")
    return df.drop(columns=["mean_jun","p10"])


def build_state_features(start, end, states, base_year):
    W = build_weather_table(start, end, states)
    Y = get_soy_yield_state(start, end, states)
    A = get_soy_acres_harvested_state(start, end, states)
    df = W.merge(Y, on=["state","year"]).merge(A, on=["state","year"])
    df = add_shortfall(df)
    df["trend"] = df["year"] - base_year
    df["dummy_2003"] = (df["year"]==2003).astype(int)
    cols = ["state","year","yield_bu_acre","trend","jun_shortfall","temp_JA","prec_JA","prec_JA_sq","dummy_2003","acres_harvested"]
    print("ok build state feature")
    return df[cols]


def aggregate_national(df_state, method="weighted"):
    rows = []
    for year, grp in df_state.groupby("year"):
        w = grp["acres_harvested"] if method=="weighted" else pd.Series(1.0, index=grp.index)
        def wavg(s): return np.average(s, weights=w)
        rows.append({
            "year": year,
            "yield_bu_acre": wavg(grp["yield_bu_acre"]),
            "trend": wavg(grp["trend"]),
            "jun_shortfall": wavg(grp["jun_shortfall"]),
            "temp_JA": wavg(grp["temp_JA"]),
            "prec_JA": wavg(grp["prec_JA"]),
            "prec_JA_sq": wavg(grp["prec_JA_sq"]),
            "dummy_2003": int(year==2003),
            "acres_harvested": grp["acres_harvested"].sum()
        })
        print("ok aggregation antiont")
    return pd.DataFrame(rows).sort_values("year")


def main():
    ensure_dirs()
    args = parse_args()
    end = args.end or pd.Timestamp.today().year
    states = [s.strip().upper() for s in args.states.split(",")]
    print(f"Building data for {states}, {args.start}-{end}")
    df_state = build_state_features(args.start, end, states, args.base_year)
    df_state.to_csv("data/processed/waob_features_state.csv", index=False)
    df_nat = aggregate_national(df_state, args.national_method)
    df_nat.to_csv("data/processed/waob_features_national.csv", index=False)
    print(df_nat.tail())


if __name__ == "__main__":
    main()
