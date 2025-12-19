import streamlit as st
import pandas as pd
import numpy as np

# ------------------------
# Page Setup
# ------------------------
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
def sample_size_penalty(routes, league_lead, scale):
    pct = routes / league_lead
    if pct >= 0.75:
        return 1.0
    return max(0.5, pct / 0.75 * scale)

# ------------------------
# Core Model
# ------------------------
def compute_model(wr_df, def_df, league_lead, sample_scaling, coverage_weight):

    results = []

    for _, row in wr_df.iterrows():

        # ---- Skip invalid rows ----
        if row["routes_played"] <= 0 or row["base_yprr"] <= 0:
            continue

        opp = row["opponent"]

        # ---- SAFE OPPONENT LOOKUP (DROP-IN FIX) ----
        if pd.isna(opp) or opp not in def_df.index:
            continue

        defense = def_df.loc[opp]
        base = row["base_yprr"]

        # ---- Sample Size Penalty ----
        sample_pen = sample_size_penalty(
            row["routes_played"], league_lead, sample_scaling
        )

        # ---- Expected YPRR vs Coverage ----
        cov_yprr = (
            defense["man_pct"] * row["yprr_man"]
            + defense["zone_pct"] * row["yprr_zone"]
        )

        safety_adj = (
            defense["onehigh_pct"] * row["yprr_1high"]
            + defense["twohigh_pct"] * row["yprr_2high"]
        )

        blitz_adj = (
            defense["blitz_pct"] * row["yprr_blitz"]
            + defense["noblitz_pct"] * row["yprr_standard"]
        )

        # Normalize to base YPRR
        coverage_ratio = (cov_yprr / base) * (safety_adj / base) * (blitz_adj / base)

        # ---- Controlled Adjustment (NO EXPLOSIONS) ----
        coverage_adjustment = 1 + coverage_weight * (coverage_ratio - 1)

        adjusted_yprr = (
            base
            * coverage_adjustment
            * sample_pen
            * row["season_route_share"]
        )

        results.append({
            "player": row["player"],
            "team": row["team"],
            "opponent": opp,
            "base_yprr": round(base, 1),
            "adjusted_yprr": round(adjusted_yprr, 1)
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df["edge"] = round(df["adjusted_yprr"] - df["base_yprr"], 1)
    df["rank"] = df["adjusted_yprr"].rank(ascending=False).astype(int)

    return df.sort_values("rank")

# ------------------------
# Run App
# ------------------------
if wr_file and def_file and matchup_file:

    wr_df = pd.read_csv(wr_file)
    def_df_raw = pd.read_csv(def_file)
    matchup_df = pd.read_csv(matchup_file)

    # ---- Detect defense team column automatically ----
    possible_team_cols = ["team", "defense", "def_team", "abbr"]
    team_col = None

    for col in possible_team_cols:
        if col in def_df_raw.columns:
            team_col = col
            break

    if team_col is None:
        st.error(
            "Defense CSV must contain a team column (e.g. 'team', 'defense', 'def_team')."
        )
        st.stop()

    def_df = def_df_raw.set_index(team_col)

    # ---- Merge weekly matchups ----
    wr_df = wr_df.merge(matchup_df, on="team", how="left")

    # ---- DEBUG WARNING PANEL ----
    missing_defs = set(wr_df["opponent"].dropna()) - set(def_df.index)

    if missing_defs:
        st.warning(
            f"Missing defense data for: {', '.join(sorted(missing_defs))}"
        )

    # ---- Run Model ----
    results = compute_model(
        wr_df,
        def_df,
        league_lead_routes,
        sample_scaling,
        coverage_weight
    )

    if results.empty:
        st.warning("No valid player projections were generated.")
        st.stop()

    # ------------------------
    # Display Results
    # ------------------------
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

