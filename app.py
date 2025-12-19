import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NFL WR YPRR Matchup Model", layout="wide")
st.title("NFL WR YPRR Matchup Model (Current Season)")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data")

wr_file = st.sidebar.file_uploader("Upload WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Upload Defense Data CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Upload Weekly Matchups CSV", type="csv")

league_lead_routes = st.sidebar.number_input(
    "League Leader Routes Run", min_value=1, value=100
)

# ------------------------
# Sliders
# ------------------------
st.sidebar.header("Model Controls")

sample_scaling = st.sidebar.slider(
    "Sample Size Penalty Strength", 0.0, 1.0, 0.5, 0.05
)

coverage_weight = st.sidebar.slider(
    "Coverage Adjustment Strength", 0.0, 1.0, 0.4, 0.05
)

# ------------------------
# Utility Functions
# ------------------------
def safe_ratio(split, base):
    if base <= 0 or pd.isna(split):
        return 1.0
    return split / base


def sample_size_penalty(routes, league_lead, scale):
    pct = routes / league_lead
    if pct >= 0.75:
        return 1.0
    return max(0.5, pct / 0.75 * scale)


# ------------------------
# Model
# ------------------------
def compute_model(wr_df, def_df, league_lead, sample_scaling, coverage_weight):

    results = []

    for _, row in wr_df.iterrows():

        if row["routes_played"] <= 0:
            continue

        base = row["base_yprr"]

        # Sample size penalty
        sample_pen = sample_size_penalty(
            row["routes_played"], league_lead, sample_scaling
        )

        defense = def_df.loc[row["opponent"]]

        # Expected YPRR based on coverage frequencies
        expected_yprr = (
            defense["man_pct"] * row["yprr_man"]
            + defense["zone_pct"] * row["yprr_zone"]
        )

        expected_yprr *= (
            defense["onehigh_pct"] * row["yprr_1high"]
            + defense["twohigh_pct"] * row["yprr_2high"]
        ) / base

        expected_yprr *= (
            defense["blitz_pct"] * row["yprr_blitz"]
            + defense["noblitz_pct"] * row["yprr_standard"]
        ) / base

        # Controlled adjustment
        coverage_adjustment = 1 + coverage_weight * (expected_yprr - 1)

        adjusted_yprr = (
            base
            * coverage_adjustment
            * sample_pen
            * row["season_route_share"]
        )

        results.append({
            "player": row["player"],
            "team": row["team"],
            "opponent": row["opponent"],
            "base_yprr": round(base, 1),
            "adjusted_yprr": round(adjusted_yprr, 1)
        })

    df = pd.DataFrame(results)

    df["edge"] = round(df["adjusted_yprr"] - df["base_yprr"], 1)
    df["rank"] = df["adjusted_yprr"].rank(ascending=False).astype(int)

    return df.sort_values("rank")


# ------------------------
# Run App
# ------------------------
if wr_file and def_file and matchup_file:

    wr_df = pd.read_csv(wr_file)
    def_df = pd.read_csv(def_file).set_index("team")
    matchup_df = pd.read_csv(matchup_file)

    wr_df = wr_df.merge(matchup_df, on="team", how="left")

    results = compute_model(
        wr_df, def_df, league_lead_routes, sample_scaling, coverage_weight
    )

    st.subheader("Adjusted YPRR Rankings")
    st.dataframe(results)

    st.subheader("Targets (Top 10 Positive Edge)")
    st.dataframe(results.sort_values("edge", ascending=False).head(10))

    st.subheader("Fades (Filtered)")
    fades = results.sort_values("edge").head(10)
    fades = fades[fades["adjusted_yprr"] >= 0.3]
    st.dataframe(fades)

else:
    st.info("Upload WR, Defense, and Matchup CSV files to begin.")
