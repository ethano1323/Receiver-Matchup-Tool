import streamlit as st
import pandas as pd
import numpy as np

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="NFL WR Matchup Model", layout="wide")
st.title("NFL WR Coverage Matchup Model (Current Season)")

# ------------------------
# Upload Data
# ------------------------
st.sidebar.header("Upload Data")

wr_file = st.sidebar.file_uploader("Upload WR Data CSV", type="csv")
def_file = st.sidebar.file_uploader("Upload Defense Data CSV", type="csv")
matchup_file = st.sidebar.file_uploader("Upload Weekly Matchups CSV", type="csv")

# ------------------------
# Utility
# ------------------------
def clamp(val, low, high):
    return max(low, min(high, val))

# ------------------------
# Core Model
# ------------------------
def compute_model(wr_df, def_df):

    results = []

    for _, row in wr_df.iterrows():

        # ---- Basic row validation ----
        if row["routes_played"] <= 0 or row["base_yprr"] <= 0:
            continue

        opp = row["opponent"]

        # ---- Safe opponent lookup ----
        if pd.isna(opp) or opp not in def_df.index:
            continue

        defense = def_df.loc[opp]
        base = row["base_yprr"]

        # ---- Expected YPRR vs Coverage ----
        expected_yprr = (
            defense["man_pct"] * row["yprr_man"]
            + defense["zone_pct"] * row["yprr_zone"]
        )

        # ---- Relative Matchup Delta ----
        raw_delta = (expected_yprr - base) / base
        raw_delta = clamp(raw_delta, -0.30, 0.30)  # cap extremes

        # ---- Matchup Edge (0–100%) ----
        matchup_edge_pct = ((raw_delta + 0.30) / 0.60) * 100

        # ---- Adjusted YPRR (Controlled) ----
        adjusted_yprr = base * (1 + raw_delta)

        results.append({
            "player": row["player"],
            "team": row["team"],
            "opponent": opp,
            "base_yprr": round(base, 1),
            "adjusted_yprr": round(adjusted_yprr, 1),
            "matchup_edge_pct": round(matchup_edge_pct, 1),
            "raw_delta_pct": round(raw_delta * 100, 1)
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    df["rank"] = df["matchup_edge_pct"].rank(ascending=False).astype(int)
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

    # ---- Debug warning for missing defenses ----
    missing_defs = set(wr_df["opponent"].dropna()) - set(def_df.index)
    if missing_defs:
        st.warning(
            f"Missing defense data for: {', '.join(sorted(missing_defs))}"
        )

    # ---- Run Model ----
    results = compute_model(wr_df, def_df)

    if results.empty:
        st.warning("No valid player projections were generated.")
        st.stop()

    # ------------------------
    # Display Results
    # ------------------------
    st.subheader("Adjusted YPRR Rankings")
    st.dataframe(results)

    st.subheader("Targets (Best Matchups)")
    st.dataframe(
        results.sort_values("matchup_edge_pct", ascending=False).head(10)
    )

    st.subheader("Fades (Worst Matchups)")
    st.dataframe(
        results.sort_values("matchup_edge_pct").head(10)
    )

else:
    st.info("Upload WR, Defense, and Matchup CSV files to begin.")

