import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NFL WR YPRR Matchup Model", layout="wide")
st.title("NFL WR YPRR Matchup Model (Current Season, Simplified)")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data")

wr_file = st.sidebar.file_uploader("Upload WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Upload Defense Data CSV", type="csv")

league_lead_routes = st.sidebar.number_input(
    "League Leader Routes Run",
    min_value=1,
    value=100
)

# ------------------------
# Sliders
# ------------------------
st.sidebar.header("Model Sliders")

sample_scaling = st.sidebar.slider(
    "Sample Size Penalty Strength",
    0.0, 2.0, 1.0, 0.05
)

coverage_weight = st.sidebar.slider(
    "Coverage Impact Strength",
    0.0, 2.0, 1.0, 0.05
)

# ------------------------
# Functions
# ------------------------

def sample_size_penalty(routes, league_lead, scale):
    pct = routes / league_lead
    if pct >= 0.75:
        return 1.0
    return max(0, (pct / 0.75) * scale)

def compute_model(wr_df, defense, league_lead, sample_scaling, coverage_weight):

    results = []

    for _, row in wr_df.iterrows():

        # Overall sample size penalty
        sample_pen = sample_size_penalty(
            row["routes_played"],
            league_lead,
            sample_scaling
        )

        # ---- MAN / ZONE (NO ROUTE WEIGHTING) ----
        man_mult = row["yprr_man"] / row["base_yprr"]
        zone_mult = row["yprr_zone"] / row["base_yprr"]

        # ---- SAFETY & BLITZ (OPTIONAL ROUTE PENALTY) ----
        onehigh_mult = (row["yprr_1high"] / row["base_yprr"])
        twohigh_mult = (row["yprr_2high"] / row["base_yprr"])
        blitz_mult = (row["yprr_blitz"] / row["base_yprr"])
        standard_mult = (row["yprr_standard"] / row["base_yprr"])

        # ---- COVERAGE FACTOR ----
        coverage_factor = (
            defense["man_pct"] * man_mult
            + defense["zone_pct"] * zone_mult
        ) * (
            defense["onehigh_pct"] * onehigh_mult
            + defense["twohigh_pct"] * twohigh_mult
        ) * (
            defense["blitz_pct"] * blitz_mult
            + defense["noblitz_pct"] * standard_mult
        )

        coverage_factor *= coverage_weight

        adjusted_yprr = (
            row["base_yprr"]
            * coverage_factor
            * sample_pen
            * row["season_route_share"]
        )

        results.append({
            "player": row["player"],
            "team": row["team"],
            "base_yprr": row["base_yprr"],
            "adjusted_yprr": adjusted_yprr
        })

    df = pd.DataFrame(results)

    league_avg = wr_df["base_yprr"].mean()
    df["edge_over_base"] = df["adjusted_yprr"] - df["base_yprr"]
    df["pct_edge_over_base"] = (df["edge_over_base"] / df["base_yprr"] * 100).round(2)
    df["edge_vs_league"] = df["adjusted_yprr"] - league_avg

    df["rank"] = df["adjusted_yprr"].rank(ascending=False, method="min").astype(int)

    return df.sort_values("rank")

# ------------------------
# Run App
# ------------------------
if wr_file is not None and def_file is not None:

    wr_df = pd.read_csv(wr_file)
    defense = pd.read_csv(def_file).iloc[0].to_dict()

    results = compute_model(
        wr_df,
        defense,
        league_lead_routes,
        sample_scaling,
        coverage_weight
    )

    st.subheader("Adjusted YPRR Rankings")
    st.dataframe(results)

    st.subheader("Targets (Top 10 Positive Edge)")
    st.dataframe(results.sort_values("edge_over_base", ascending=False).head(10))

    st.subheader("Fades (Top 10 Negative Edge)")
    st.dataframe(results.sort_values("edge_over_base").head(10))

else:
    st.info("Upload WR and Defense CSV files to begin.")
