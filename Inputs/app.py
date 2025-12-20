def compute_model_vectorized(
    wr_df,
    def_df,
    max_penalty=0.8,
    exponent=2,
    start_penalty=0.50,
    end_penalty=0.05
):
    # ------------------------
    # Base calculations
    # ------------------------
    # Filter out invalid WRs early
    wr_df = wr_df[(wr_df["base_yprr"] >= 0.4) & (wr_df["routes_played"] > 0)].copy()
    if wr_df.empty:
        return pd.DataFrame()

    # Normalize opponent column
    wr_df = wr_df[~wr_df["opponent"].isna()]
    wr_df = wr_df[wr_df["opponent"].isin(def_df.index)]
    if wr_df.empty:
        return pd.DataFrame()

    # League-average YPRR vs blitz
    league_avg_blitz = wr_df["yprr_blitz"].mean(skipna=True)
    wr_df["yprr_blitz"].fillna(league_avg_blitz, inplace=True)

    # ------------------------
    # Ratios for coverage and safety
    # ------------------------
    # Avoid division by zero
    base = wr_df["base_yprr"].replace(0, np.nan)

    coverage_cols = ["yprr_man", "yprr_zone"]
    safety_cols = ["yprr_1high", "yprr_2high", "yprr_0high"]

    # Coverage ratios
    for col in coverage_cols:
        wr_df[col + "_ratio"] = wr_df[col] / base

    # Safety ratios
    for col in safety_cols:
        wr_df[col + "_ratio"] = wr_df[col] / base

    # Blitz ratio
    wr_df["blitz_ratio"] = wr_df["yprr_blitz"] / base

    # ------------------------
    # Merge defense data
    # ------------------------
    def_cols = ["man_pct", "zone_pct", "onehigh_pct", "twohigh_pct", "zerohigh_pct", "blitz_pct"]
    wr_df = wr_df.merge(def_df[def_cols], left_on="opponent", right_index=True, how="left")

    # Normalize coverage and safety
    wr_df["coverage_component"] = wr_df["man_pct"] * wr_df["yprr_man_ratio"] + wr_df["zone_pct"] * wr_df["yprr_zone_ratio"]

    wr_df["safety_total"] = wr_df[["onehigh_pct", "twohigh_pct", "zerohigh_pct"]].sum(axis=1)
    wr_df["safety_component"] = np.where(
        wr_df["safety_total"] > 0,
        (wr_df["onehigh_pct"] * wr_df["yprr_1high_ratio"] +
         wr_df["twohigh_pct"] * wr_df["yprr_2high_ratio"] +
         wr_df["zerohigh_pct"] * wr_df["yprr_0high_ratio"]) / wr_df["safety_total"],
        0
    )

    wr_df["coverage_safety_ratio"] = (wr_df["coverage_component"] + wr_df["safety_component"]) / 2

    # Blitz component
    wr_df["blitz_component"] = wr_df["blitz_pct"] * wr_df["blitz_ratio"] + (1 - wr_df["blitz_pct"]) * 1.0

    # Expected ratio
    wr_df["expected_ratio"] = (wr_df["coverage_safety_ratio"] + wr_df["blitz_component"]) / 2

    # Adjusted YPRR
    wr_df["adjusted_yprr"] = wr_df["base_yprr"] * wr_df["expected_ratio"]

    # Raw edge
    wr_df["raw_edge"] = (wr_df["adjusted_yprr"] - wr_df["base_yprr"]) / wr_df["base_yprr"]
    wr_df["raw_edge"] = wr_df["raw_edge"].clip(-0.25, 0.25)
    wr_df["edge_score"] = (wr_df["raw_edge"] / 0.25) * 100

    # ------------------------
    # Route share and penalty
    # ------------------------
    # Percentile-based league lead for smoothing
    league_lead_routes = wr_df["routes_played"].quantile(0.95)
    wr_df["route_share"] = wr_df["routes_played"] / league_lead_routes

    # Penalty calculation (vectorized)
    conditions = [
        wr_df["route_share"] >= start_penalty,
        wr_df["route_share"] <= end_penalty
    ]
    choices = [0, max_penalty]
    wr_df["penalty"] = np.select(conditions, choices, default=max_penalty * ((start_penalty - wr_df["route_share"]) / (start_penalty - end_penalty)) ** exponent)

    wr_df["edge_score"] = wr_df["edge_score"] * (1 - wr_df["penalty"])

    # ------------------------
    # Format results
    # ------------------------
    results = wr_df[[
        "player", "team", "opponent", "route_share",
        "base_yprr", "adjusted_yprr", "edge_score"
    ]].copy()

    results.rename(columns={
        "player": "Player",
        "team": "Team",
        "opponent": "Opponent",
        "route_share": "Route Share",
        "base_yprr": "Base YPRR",
        "adjusted_yprr": "Adjusted YPRR",
        "edge_score": "Edge"
    }, inplace=True)

    # Filter qualified players if toggled
    if qualified_toggle:
        results = results[results["Route Share"] >= 0.35]

    # Rank
    results = results.sort_values("Edge", key=abs, ascending=False)
    results["Rank"] = range(1, len(results) + 1)

    return results
