# src/soy_conditions.py
import os, requests, pandas as pd
from typing import Iterable, Tuple, Optional

USDA_API_KEY = os.getenv("USDA_API_KEY") or "D1ABF2AD-362D-346E-A641-93A2FA6ED6D8"
BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
SEVEN_STATES = ("IA","IL","IN","OH","MO","MN","NE",)  

# ----------------------------- Fetch weekly -----------------------------
def fetch_soy_condition_states(year_from: int, year_to: int,
                               states: SEVEN_STATES) -> pd.DataFrame:
    q = {
        "key": USDA_API_KEY, "page_size": 50000,
        "source_desc": "SURVEY",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "CONDITION",
        "agg_level_desc": "STATE",
        "year__GE": int(year_from), "year__LE": int(year_to),
        "state_alpha": states,
    }
    out, page = [], 1
    while True:
        r = requests.get(BASE, params={**q, "page": page}, timeout=60)
        if r.status_code != 200:
            print("DEBUG URL:", r.url, "\nDEBUG BODY:", r.text[:500])
            r.raise_for_status()
        data = r.json().get("data", [])
        if not data: break
        out.extend(data)
        if len(data) < q["page_size"]: break
        page += 1

    df = pd.DataFrame(out)
    need = ["year","reference_period_desc","state_alpha","unit_desc","Value"]
    miss = set(need).difference(df.columns)
    if miss: raise ValueError(f"Colonnes manquantes: {sorted(miss)}")

    df = (df[need]
          .rename(columns={"Value":"pct","reference_period_desc":"week",
                           "state_alpha":"state","unit_desc":"condition"}))
    df["week"] = (df["week"].astype(str).str.replace("WEEK #","",regex=False)
                             .str.extract(r"(\d+)").astype(int))
    df["condition"] = df["condition"].astype(str).str.replace("PCT ","",regex=False).str.strip()
    df["pct"] = pd.to_numeric(df["pct"].astype(str).str.replace(",","",regex=False), errors="coerce")
    return df

# ----------------------------- Weekly → compact weekly -------------
def to_compact_weekly(df: pd.DataFrame) -> pd.DataFrame:
    cond_map = {"EXCELLENT":"GE","GOOD":"GE","FAIR":"FAIR","POOR":"PVP","VERY POOR":"PVP"}
    agg = (df.assign(cond_grp=df["condition"].map(cond_map))
             .groupby(["year","state","week","cond_grp"], as_index=False)["pct"].mean()
             .pivot(index=["year","state","week"], columns="cond_grp", values="pct")
             .reset_index().fillna(0).sort_values(["year","state","week"]))
    for c in ["GE","FAIR","PVP"]:
        if c not in agg.columns: agg[c] = 0.0
    return agg

def build_condition_yearly_features(weekly_compact: pd.DataFrame) -> pd.DataFrame:
    w = weekly_compact.copy()
    for c in ["GE","FAIR","PVP"]:
        if c not in w.columns: w[c] = 0.0
    w["cond_index"] = (5*w["GE"] + 3*w["FAIR"] + 1*w["PVP"]) / 100.0

    ja = w[w["week"].between(27, 35)]  
    feats = (ja.groupby(["year","state"])
               .agg(gex_JA_mean=("GE","mean"),
                    gex_JA_min=("GE","min"),
                    fair_JA_mean=("FAIR","mean"),
                    pvp_JA_max=("PVP","max"),
                    cond_index_JA_mean=("cond_index","mean"))
               .reset_index())

    w31 = (w[w["week"].eq(31)]
           .groupby(["year","state"])["GE"].mean().rename("gex_week31").reset_index())
    feats = feats.merge(w31, on=["year","state"], how="left")

    wk = w[w["week"].isin([24,35])].pivot_table(index=["year","state"], columns="week", values="GE")
    wk = wk.rename(columns={24:"GE_start",35:"GE_end"}).reset_index()
    wk["gex_trend"] = wk.get("GE_end",0).fillna(0) - wk.get("GE_start",0).fillna(0)
    feats = feats.merge(wk[["year","state","gex_trend"]], on=["year","state"], how="left").fillna(0)

    return feats


def get_soy_condition_features(year_from: int=1988, year_to: int=2025,
                               states: Iterable[str]=SEVEN_STATES,
                               save_path: Optional[str]=None
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (weekly_compact, annual_features). Sauve en Parquet/CSV si save_path est fourni.
    """
    weekly_raw = fetch_soy_condition_states(year_from, year_to, states)
    weekly_compact = to_compact_weekly(weekly_raw)
    annual_features = build_condition_yearly_features(weekly_compact)

    if save_path:
        if save_path.lower().endswith(".parquet"):
            annual_features.to_parquet(save_path, index=False)
        elif save_path.lower().endswith(".csv"):
            annual_features.to_csv(save_path, index=False)
        else:
            annual_features.to_parquet(save_path + ".parquet", index=False)
    return weekly_compact, annual_features

if __name__ == "__main__":
    wk, feats = get_soy_condition_features(1887, 2024, SEVEN_STATES, save_path="data/processed/soy_conditions_features.parquet")
    print("Weekly shape:", wk.shape)
    print("Annual shape:", feats.shape)
    print(feats.head())