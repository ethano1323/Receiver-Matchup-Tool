import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL Receiver Matchup Model", layout="wide")
st.markdown("<h1 style='color:#ff6f6f'>Receiver Matchup Weekly Model</h1>", unsafe_allow_html=True)

# ------------------------
# Default Data Paths
# ------------------------
DEFAULT_WR_PATH = "data/standard_wr_data.csv"
DEFAULT_DEF_PATH = "data/standard_def_data.csv"
DEFAULT_MATCHUP_PATH = "data/standard_matchup_data.csv"
DEFAULT_BLITZ_PATH = "data/standard_blitz_data.csv"

# ------------------------
# Upload Data (Optional Overrides)
# ------------------------
st.sidebar.header("Control Panel")

wr_file = st.sidebar.file_uploader("WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Defense Tendencies CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Weekly Matchups CSV", type="csv")
blitz_file = st.sidebar.file_uploader("WR Blitz YPRR CSV", type="csv")

# ------------------------
# Route-share filter toggles
# ------------------------
qualified_toggle_35 = st.sidebar.checkbox("Show only players ≥35% route share")
qualified_toggle_20 = st.sidebar.checkbox("Show only players ≥20% route share")

# ------------------------
# Load Data
# ------------------------
try:
    wr_df = pd.read_csv(wr_file) if wr_file else pd.read_csv(DEFAULT_WR_PATH)
    def_df_raw = pd.read_csv(def_file) if def_file else pd.read_csv(DEFAULT_DEF_PATH)
    matchup_df = pd.read_csv(matchup_file) if matchup_file else pd.read_csv(DEFAULT_MATCHUP_PATH)
    blitz_df = pd.read_csv(blitz_file) if blitz_file else pd.read_csv(DEFAULT_BLITZ_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ------------------------
# Normalize player names
# ------------------------
def normalize_name(name):
    return str(name).lower().replace(".", "").replace(" jr", "").replace(" iii", "").strip()

wr_df["player_norm"] = wr_df["player"].apply(normalize_name)
blitz_df["player_norm"] = blitz_df["player"].apply(normalize_name)

wr_df = wr_df.merge(
    blitz_df[["player_norm", "yprr_blitz"]],
    on="player_norm",
    how="left"
)

# ------------------------
# Defense Data
# ------------------------
for col in ["team", "defense", "def_team", "abbr"]:
    if col in def_df_raw.columns:
        def_df = def_df_raw.set_index(col)
        break
else:
    st.error("Defense CSV must include a team column.")
    st.stop()

for col in [
    "man_pct", "zone_pct",
    "onehigh_pct", "twohigh_pct", "zerohigh_pct",
    "blitz_pct"
]:
    def_df[col] /= 100.0

# ------------------------
# Merge matchups
# ------------------------
wr_df = wr_df.merge(matchup_df, on="team", how="left")

# ------------------------
# League averages (unchanged)
# ------------------------
league_avg_man = def_df["man_pct"].mean()
league_avg_zone = def_df["zone_pct"].mean()
league_avg_1high = def_df["onehigh_pct"].mean()
league_avg_2high = def_df["twohigh_pct"].mean()
league_avg_0high = def_df["zerohigh_pct"].mean()

# ------------------------
# Core Model
# ------------------------
def compute_model(
    wr_df,
    def_df,
    max_penalty=0.6,
    exponent=2,
    start_penalty=30,
    end_penalty=5,
    deviation_boost=0.25
):
    REG_K = 20
    MIN_RATIO = 0.6
    MAX_RATIO = 1.6

    results = []

    for _, row in wr_df.iterrows():
        base = row["base_yprr"]
        routes = row["routes_played"]

        if base < 0.4 or routes <= 0:
            continue

        opponent = row["opponent"]
        if pd.isna(opponent) or opponent not in def_df.index:
            continue

        defense = def_df.loc[opponent]

        # ------------------------
        # Player-relative regression helper
        def regressed_ratio(split_yprr):
            if pd.isna(split_yprr):
                split_yprr = base
            effective = (
                split_yprr * routes +
                base * REG_K
            ) / (routes + REG_K)
            ratio = effective / base
            return np.clip(ratio, MIN_RATIO, MAX_RATIO)

        # ------------------------
        # Ratios (REGRESSED + CLAMPED)
        man_ratio = regressed_ratio(row["yprr_man"])
        zone_ratio = regressed_ratio(row["yprr_zone"])
        onehigh_ratio = regressed_ratio(row["yprr_1high"])
        twohigh_ratio = regressed_ratio(row["yprr_2high"])
        zerohigh_ratio = regressed_ratio(row["yprr_0high"])
        blitz_ratio = regressed_ratio(row.get("yprr_blitz", base))

        # ------------------------
        # System A
        coverage_component = (
            defense["man_pct"] * man_ratio +
            defense["zone_pct"] * zone_ratio
        )
        total_coverage = defense["man_pct"] + defense["zone_pct"]

        safety_component = (
            defense["onehigh_pct"] * onehigh_ratio +
            defense["twohigh_pct"] * twohigh_ratio +
            defense["zerohigh_pct"] * zerohigh_ratio
        )
        total_safety = (
            defense["onehigh_pct"] +
            defense["twohigh_pct"] +
            defense["zerohigh_pct"]
        )

        if total_safety > 0:
            safety_component /= total_safety

        if total_coverage + total_safety > 0:
            systemA_ratio = (
                coverage_component * total_coverage +
                safety_component * total_safety
            ) / (total_coverage + total_safety)
        else:
            systemA_ratio = (coverage_component + safety_component) / 2

        # ------------------------
        # System B (UNCHANGED)
        coverage_dev = abs(defense["man_pct"] - league_avg_man) + abs(defense["zone_pct"] - league_avg_zone)
        safety_dev = abs(defense["onehigh_pct"] - league_avg_1high) + abs(defense["twohigh_pct"] - league_avg_2high) + abs(defense["zerohigh_pct"] - league_avg_0high)

        if coverage_dev + safety_dev > 0:
            coverage_weight_dev = coverage_dev / (coverage_dev + safety_dev)
            safety_weight_dev = safety_dev / (coverage_dev + safety_dev)
        else:
            coverage_weight_dev = 0.5
            safety_weight_dev = 0.5

        systemB_ratio = (
            coverage_component * coverage_weight_dev +
            safety_component * safety_weight_dev
        )

        # ------------------------
        # Hybrid
        final_ratio = (
            systemA_ratio * (1 - deviation_boost) +
            systemB_ratio * deviation_boost
        )

        adjusted_yprr = base * ((final_ratio + blitz_ratio) / 2)
        raw_edge = (adjusted_yprr - base) / base
        edge_score = raw_edge * 100  # NO CAP

        # ------------------------
        # Route-share penalty (UNCHANGED)
        route_share = row.get("route_share", 0)

        if route_share >= start_penalty:
            penalty = 0
        elif route_share <= end_penalty:
            penalty = max_penalty
        else:
            penalty = max_penalty * ((start_penalty - route_share) / (start_penalty - end_penalty)) ** exponent

        edge_score *= (1 - penalty)

        results.append({
            "Player": row["player"],
            "Tm": row["team"],
            "Vs.": opponent,
            "Route (%)": route_share,
            "Base YPRR": round(base, 2),
            "Adj. YPRR": round(adjusted_yprr, 2),
            "Matchup (+/-)": round(edge_score * (1 - deviation_boost), 1),
            "Deviation": round(edge_score * deviation_boost, 1),
            "Edge": round(edge_score, 1)
        })

    df = pd.DataFrame(results)

    if qualified_toggle_35:
        df = df[df["Route (%)"] >= 35]
    elif qualified_toggle_20:
        df = df[df["Route (%)"] >= 20]

    df = df.reindex(df["Edge"].abs().sort_values(ascending=False).index)
    df["Rk"] = range(1, len(df) + 1)

    return df

