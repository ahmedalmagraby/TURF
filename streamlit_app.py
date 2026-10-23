"""TURF Analysis Automation Tool — Streamlit front end.

All maths lives in ``turf_core`` (no Streamlit dependency, unit-tested). This file only
handles input, caching, and presentation.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import turf_core as tc

logger = logging.getLogger(__name__)

MAX_PORTFOLIO_SIZE = 12          # largest portfolio size offered in the UI
MAX_ITEM_COLUMNS = 200           # guard against selecting an entire wide survey file
DIMINISHING_RETURNS_PCT = 2.0    # an extra item adding less than this is "diminishing"
RESULT_PREFIX = "turf_"          # every session-state key holding analysis results

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="TURF Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 TURF Analysis Automation Tool")
st.markdown("""
**Total Unduplicated Reach and Frequency (TURF) Analysis** helps identify the optimal
combination of items that maximizes audience reach while minimizing overlap.
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📖 Methodology & Guide")

    st.subheader("What is TURF?")
    st.markdown("""
    TURF analysis identifies the best combination of items (products, features, messages)
    that reaches the maximum number of unique respondents.
    """)

    st.subheader("Key Metrics")
    st.markdown("""
    - **Reach (%)**: % of unique respondents who selected ≥1 item in the combination
    - **Frequency**: Total selections across all items in the combination
    - **Incremental Reach**: Additional reach gained by adding one more item
    """)

    st.subheader("Analysis Modes")
    st.markdown("""
    - **Full TURF**: Evaluates every combination of size k to find the optimum. When the
      search space is too large, a random sample plus the greedy solution is evaluated.
    - **Greedy Sequential**: Builds the portfolio one item at a time, always adding
      whichever item gains the most new respondents — fast and interpretable, but a
      heuristic that is not guaranteed to find the best possible combination
    """)

    st.subheader("Data Format")
    st.markdown("""
    - **Rows**: Each row = one respondent
    - **Columns**: Each column = one item/product/feature
    - **Values**: 1 (selected) or 0 (not selected). `True`/`False` and `Yes`/`No` also work;
      blanks count as 0

    ⚠️ Exclude ID columns and demographic variables!
    """)

    st.subheader("How to Use")
    st.markdown("""
    1. Choose data source (Upload or Simulate)
    2. Optionally set Must-Include / Must-Exclude constraints
    3. Select portfolio size (k)
    4. Click **Run TURF Analysis**
    5. Review reach curve, top combinations, and greedy path
    """)


# ── Cached wrappers (keep the maths in turf_core free of Streamlit) ───────────
# Bounded caches (max_entries / ttl) keep memory in check and make sure uploaded
# respondent data does not linger in the server process indefinitely.
@st.cache_data(show_spinner=False, max_entries=8)
def cached_simulated_data(n_respondents: int, n_items: int) -> pd.DataFrame:
    return tc.simulate_data(n_respondents, n_items)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_sheet_names(data: bytes) -> list[str]:
    return tc.list_excel_sheets(data)


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def cached_read_table(data: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    return tc.read_table(data, filename, sheet)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_search(df: pd.DataFrame, k: int, must_include: tuple, must_exclude: tuple):
    return tc.calculate_turf(df, k, must_include, must_exclude)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_greedy(df: pd.DataFrame, max_k: int, must_include: tuple, must_exclude: tuple):
    return tc.greedy_turf(df, max_k, must_include, must_exclude)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_optimal_by_size(df: pd.DataFrame, max_k: int, must_include: tuple, must_exclude: tuple):
    return tc.optimal_by_size(df, max_k, must_include, must_exclude)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_individual_reach(df: pd.DataFrame, must_exclude: tuple):
    return tc.individual_reach(df, must_exclude)


# ── Small helpers ─────────────────────────────────────────────────────────────
def names(items) -> str:
    """Comma-separated item names, safe to embed in Markdown."""
    return ", ".join(f"`{str(i).replace('`', chr(39))}`" for i in items)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    # utf-8-sig so Excel opens non-ASCII item names correctly
    return frame.to_csv(index=False).encode("utf-8-sig")


def clear_results() -> None:
    for key in [k for k in st.session_state if k.startswith(RESULT_PREFIX)]:
        del st.session_state[key]


# ══════════════════════════════════════════════════════════════════════════════
#  UI — Data source
# ══════════════════════════════════════════════════════════════════════════════
def render_upload() -> pd.DataFrame | None:
    """File upload + column selection. Returns the cleaned 0/1 matrix, or None."""
    st.info("📁 Upload your binary data file (CSV or Excel)")
    uploaded = st.file_uploader(
        "Choose a file",
        type=[s.lstrip(".") for s in tc.SUPPORTED_SUFFIXES],
        help="A CSV or Excel file with one row per respondent and one 0/1 column per item.",
    )
    if uploaded is None:
        return None

    data = uploaded.getvalue()
    sheet = None
    if uploaded.name.lower().endswith(tc.EXCEL_SUFFIXES):
        try:
            sheets = cached_sheet_names(data)
        except tc.DataError as exc:
            st.error(f"❌ {exc}")
            return None
        sheet = st.selectbox("Worksheet", sheets) if len(sheets) > 1 else sheets[0]

    try:
        raw = cached_read_table(data, uploaded.name, sheet)
    except tc.DataError as exc:
        st.error(f"❌ {exc}")
        return None
    st.success(f"✅ File loaded: {len(raw):,} rows × {len(raw.columns)} columns")

    st.subheader("📋 Select Columns to Analyze")
    st.markdown("""
    **Important:** Only select columns that represent items/products/features.
    ⚠️ Exclude respondent IDs, demographics, timestamps.
    """)
    all_columns = raw.columns.tolist()
    id_columns = [c for c in all_columns if tc.is_id_column(c)]
    binary_columns = [c for c in all_columns if c not in id_columns and tc.is_binary_like(raw[c])]
    default_columns = binary_columns or [c for c in all_columns if c not in id_columns]
    if id_columns:
        st.warning(f"⚠️ Likely ID/metadata columns excluded by default: {names(id_columns)}")
    if binary_columns:
        st.caption(
            f"{len(binary_columns)} column(s) containing only 0/1 (or yes/no) values were pre-selected."
        )

    selected = st.multiselect(
        "Select columns to include in TURF analysis:",
        options=all_columns,
        default=default_columns,
        help="Choose only item/product columns.",
    )
    if len(selected) < 2:
        st.error("❌ Please select at least 2 columns for analysis.")
        return None
    if len(selected) > MAX_ITEM_COLUMNS:
        st.error(f"❌ Too many columns selected ({len(selected)}). The limit is {MAX_ITEM_COLUMNS}; "
                 "deselect demographics and other non-item columns.")
        return None

    cleaned = tc.clean_binary_data(raw, selected)
    for message in cleaned.warnings:
        st.warning(f"⚠️ {message}")
    if cleaned.error:
        st.error(f"❌ {cleaned.error}")
        return None
    st.success(f"✅ Ready: {len(cleaned.data):,} respondents × {len(cleaned.data.columns)} items")
    return cleaned.data


st.header("1️⃣ Data Source Selection")

data_source = st.radio(
    "Choose your data source:",
    options=["Use Simulated Data", "Upload Own Data"],
    help="Select simulated data to test the tool, or upload your own CSV/Excel file.",
)

df: pd.DataFrame | None = None

if data_source == "Use Simulated Data":
    st.info("📊 Using simulated data for demonstration purposes.")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        n_respondents = st.slider("Number of Respondents", 50, 500, 200, 50)
    with sim_col2:
        n_items = st.slider("Number of Items", 5, 20, 12, 1)
    df = cached_simulated_data(n_respondents, n_items)
    st.success(f"✅ Generated {n_respondents} respondents × {n_items} items")
else:
    df = render_upload()

# ── Data preview ──────────────────────────────────────────────────────────────
if df is not None:
    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(10), width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Respondents", f"{len(df):,}")
        c2.metric("Items", len(df.columns))
        c3.metric("Overall Selection Rate", f"{df.to_numpy().mean() * 100:.1f}%")
        st.subheader("Item Selection Rates")
        item_rates = (df.mean() * 100).sort_values(ascending=False)
        fig_prev = px.bar(
            x=item_rates.index, y=item_rates.values,
            labels={"x": "Item", "y": "Selection Rate (%)"},
            title="Individual Item Selection Rates",
        )
        st.plotly_chart(fig_prev, width="stretch")
else:
    # Nothing to analyze (e.g. the file was removed): never show results from earlier data.
    clear_results()

# ══════════════════════════════════════════════════════════════════════════════
#  UI — Analysis configuration
# ══════════════════════════════════════════════════════════════════════════════
if df is not None:
    st.header("2️⃣ Analysis Configuration")
    all_items = df.columns.tolist()

    with st.expander("🔒 Item Constraints (Optional)", expanded=False):
        st.markdown("""
        **Must-Include**: items that are always in every portfolio (e.g., a flagship product).
        **Must-Exclude**: items that should never appear (e.g., discontinued lines).
        """)
        col_mi, col_me = st.columns(2)
        with col_mi:
            must_include = st.multiselect(
                "✅ Must-Include items:", options=all_items, default=[],
                help="These items are locked into every combination.",
            )
        with col_me:
            must_exclude = st.multiselect(
                "🚫 Must-Exclude items:",
                options=[it for it in all_items if it not in must_include], default=[],
                help="These items will never appear in any combination.",
            )

    n_locked = len(must_include)
    n_free = len([it for it in all_items if it not in must_include and it not in must_exclude])
    # k counts must-include items too and always leaves at least one free slot.
    min_k = max(2, n_locked + 1)
    max_k = min(MAX_PORTFOLIO_SIZE, n_locked + n_free)

    if n_free < 1 or min_k > max_k:
        st.error("❌ Too many constraints — at least two items must remain, including one "
                 f"that is not locked in (portfolio sizes up to {MAX_PORTFOLIO_SIZE} are supported).")
    else:
        col_k, col_space = st.columns([2, 1])
        with col_k:
            if min_k == max_k:
                k = min_k
                st.info(f"Portfolio size is fixed at k = {k} with the current items and constraints.")
            else:
                k = st.slider(
                    "Portfolio Size (k)", min_value=min_k, max_value=max_k,
                    value=max(min_k, min(3, max_k)),
                    help="Total number of items in each combination (includes must-include items)",
                )
        with col_space:
            st.metric("Search Space", f"{math.comb(n_free, k - n_locked):,} combos")

        mi_key, me_key = tuple(must_include), tuple(must_exclude)
        # The fingerprint covers the data itself, not just its shape, so results are
        # flagged stale if a different file with the same layout is loaded.
        run_fp = tc.fingerprint(df, k, sorted(must_include), sorted(must_exclude))

        if st.button("🚀 Run TURF Analysis", type="primary", width="stretch"):
            try:
                with st.spinner("Running TURF analysis…"):
                    ind_reach_df = cached_individual_reach(df, me_key)
                    search = cached_search(df, k, mi_key, me_key)
                    greedy = cached_greedy(df, max_k, mi_key, me_key)
                    opt_size_df = cached_optimal_by_size(df, max_k, mi_key, me_key)
            except ValueError as exc:
                st.error(f"❌ {exc}")
            except Exception:  # noqa: BLE001 - last line of defence for the UI
                logger.exception("TURF analysis failed")
                st.error("❌ The analysis failed unexpectedly. Try fewer items or a smaller file; "
                         "if it keeps happening, please report the problem.")
            else:
                st.session_state.update({
                    f"{RESULT_PREFIX}results": search.results,
                    f"{RESULT_PREFIX}is_sampled": search.is_sampled,
                    f"{RESULT_PREFIX}theo_combos": search.theoretical_combos,
                    f"{RESULT_PREFIX}coverage": search.coverage_pct,
                    f"{RESULT_PREFIX}k": k,
                    f"{RESULT_PREFIX}n": len(df),
                    f"{RESULT_PREFIX}greedy": greedy,
                    f"{RESULT_PREFIX}fingerprint": run_fp,
                    f"{RESULT_PREFIX}must_include": list(must_include),
                    f"{RESULT_PREFIX}individual_reach": ind_reach_df,
                    f"{RESULT_PREFIX}optimal_by_size": opt_size_df,
                })

        if (f"{RESULT_PREFIX}results" in st.session_state
                and st.session_state.get(f"{RESULT_PREFIX}fingerprint") != run_fp):
            st.warning("⚠️ The data or configuration has changed since the last analysis. "
                       "Click **Run TURF Analysis** to refresh results.")

# ══════════════════════════════════════════════════════════════════════════════
#  UI — Results
# ══════════════════════════════════════════════════════════════════════════════
if f"{RESULT_PREFIX}results" in st.session_state:
    ss = st.session_state
    res_df: pd.DataFrame = ss[f"{RESULT_PREFIX}results"]
    is_sampled: bool = ss[f"{RESULT_PREFIX}is_sampled"]
    theo_combos: int = ss[f"{RESULT_PREFIX}theo_combos"]
    coverage: float = ss[f"{RESULT_PREFIX}coverage"]
    stored_k: int = ss[f"{RESULT_PREFIX}k"]
    n_resp: int = ss[f"{RESULT_PREFIX}n"]
    greedy: tc.GreedyResult = ss[f"{RESULT_PREFIX}greedy"]
    locked_items: list[str] = ss[f"{RESULT_PREFIX}must_include"]
    ind_reach_df: pd.DataFrame = ss[f"{RESULT_PREFIX}individual_reach"]
    opt_size_df: pd.DataFrame = ss[f"{RESULT_PREFIX}optimal_by_size"]
    mi_count = len(locked_items)

    st.header("3️⃣ Analysis Results")

    if is_sampled:
        st.warning(
            f"⚠️ Search space too large ({theo_combos:,} combinations). Evaluated "
            f"{len(res_df):,} randomly sampled combinations ({coverage}% of the space) plus the "
            "greedy solution. The best portfolio below is not guaranteed to be the true optimum."
        )
    else:
        st.info(f"ℹ️ Exhaustive search: all {theo_combos:,} combination(s) evaluated (k={stored_k}).")

    tab_individual, tab_optimal_size, tab_full, tab_greedy, tab_curve = st.tabs(
        ["📍 Individual Reach", "🔢 Optimal by Size", "🏆 Top Combinations",
         "🪜 Greedy Path", "📈 Reach Curve"]
    )

    # ── Tab: Individual reach ─────────────────────────────────────────────────
    with tab_individual:
        st.subheader("📍 Individual Touchpoint Reach")
        st.markdown("""
        Standalone reach of **each individual touchpoint** — the percentage of respondents
        reached by that item alone, independent of any other items.
        """)
        if len(ind_reach_df) == 0:
            st.info("No items to display (all may be excluded).")
        else:
            top3 = ind_reach_df.head(3)
            metric_cols = st.columns(len(top3))
            for col, (_, row) in zip(metric_cols, top3.iterrows(), strict=True):
                col.metric(
                    label=row["Item"],
                    value=f"{row['Reach (%)']:.1f}%",
                    help=f"{row['Reach (Count)']} of {n_resp} respondents",
                )
            st.dataframe(ind_reach_df, width="stretch", hide_index=True)

            chart_df = ind_reach_df.iloc[::-1]  # ascending order for a horizontal bar
            fig_ind = go.Figure(go.Bar(
                y=chart_df["Item"], x=chart_df["Reach (%)"], orientation="h",
                text=chart_df["Reach (%)"].map(lambda x: f"{x:.1f}%"),
                textposition="auto",
                marker=dict(color=chart_df["Reach (%)"], colorscale="Teal", showscale=False),
                hovertemplate="<b>%{y}</b><br>Reach: %{x:.2f}%<extra></extra>",
            ))
            fig_ind.update_layout(
                title="Standalone Reach per Touchpoint",
                xaxis_title="Reach (%)", yaxis_title="",
                height=max(350, len(ind_reach_df) * 30),
                margin=dict(l=10, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_ind, width="stretch")
            st.download_button(
                "⬇️ Download Individual Reach as CSV", data=csv_bytes(ind_reach_df),
                file_name="turf_individual_reach.csv", mime="text/csv",
            )

    # ── Tab: Optimal by portfolio size ────────────────────────────────────────
    with tab_optimal_size:
        st.subheader("🔢 Optimal Combinations by Portfolio Size")
        st.markdown("""
        The **best combination** at each portfolio size (1, 2, 3, …) and how much
        **incremental reach** is gained by adding one more touchpoint to the optimal set.
        Use this to decide the ideal portfolio size — where the incremental gain no longer
        justifies the cost of another item. Sizes marked *Greedy approx.* were too large to
        search exhaustively.
        """)
        if len(opt_size_df) == 0:
            st.info("No data to display.")
        else:
            st.dataframe(opt_size_df, width="stretch", hide_index=True)

            inc = opt_size_df["Incremental Reach (%)"]
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=opt_size_df["Size"], y=opt_size_df["Reach (%)"],
                mode="lines+markers+text", name="Reach (%)",
                text=opt_size_df["Reach (%)"].map(lambda x: f"{x:.1f}%"),
                textposition="top center",
                line=dict(color="steelblue", width=3), marker=dict(size=9),
            ))
            fig_opt.add_trace(go.Bar(
                x=opt_size_df["Size"], y=inc, name="Incremental Reach (%)",
                opacity=0.45, marker_color="coral", yaxis="y2",
                text=inc.map(lambda x: f"+{x:.1f}%"), textposition="outside",
                hovertemplate="Size %{x}: +%{y:.2f}%<extra></extra>",
            ))
            fig_opt.update_layout(
                xaxis=dict(title="Portfolio Size", tickmode="array",
                           tickvals=opt_size_df["Size"].tolist()),
                yaxis=dict(title="Cumulative Reach (%)", range=[0, 105]),
                yaxis2=dict(title="Incremental Reach (%)", overlaying="y", side="right",
                            showgrid=False, range=[0, max(inc.max() * 1.5, 10)]),
                legend=dict(x=0.01, y=0.99), height=500,
                title="Optimal Reach by Portfolio Size with Incremental Gains",
                hovermode="x unified",
            )
            st.plotly_chart(fig_opt, width="stretch")

            rec_size = tc.diminishing_returns_size(
                opt_size_df["Size"].tolist(), inc.tolist(), DIMINISHING_RETURNS_PCT)
            if rec_size is not None:
                st.info(f"💡 **Diminishing returns detected:** incremental reach drops below "
                        f"{DIMINISHING_RETURNS_PCT:g}% after portfolio size {rec_size}. "
                        f"Consider {rec_size} items as the sweet spot.")
            st.download_button(
                "⬇️ Download Optimal-by-Size as CSV", data=csv_bytes(opt_size_df),
                file_name="turf_optimal_by_size.csv", mime="text/csv",
            )

    # ── Tab: Top combinations ─────────────────────────────────────────────────
    with tab_full:
        if len(res_df) == 0:
            st.error("No valid combinations found with current constraints.")
        else:
            top = res_df.iloc[0]
            st.subheader("🏆 Best Portfolio")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Reach", f"{top['Reach (%)']:.1f}%")
            m2.metric("Respondents Reached", f"{top['Reach (Count)']:,} / {n_resp:,}")
            m3.metric("Not Reached", f"{100 - top['Reach (%)']:.1f}%")
            m4.metric("Total Selections", f"{top['Frequency']:,}")
            st.markdown(f"**Portfolio:** {names(top['Combination'].split(' + '))}")

            st.subheader("📊 Top 20 Combinations")
            st.dataframe(res_df.head(20), width="stretch", hide_index=True)

            top_n = min(15, len(res_df))
            top_rows = res_df.head(top_n).copy()
            top_rows["Label"] = top_rows["Combination"].apply(
                lambda s: s if len(s) <= 45 else s[:45] + "…")
            top_rows = top_rows.iloc[::-1]
            fig = go.Figure(go.Bar(
                y=top_rows["Label"], x=top_rows["Reach (%)"], orientation="h",
                text=top_rows["Reach (%)"].map(lambda x: f"{x:.1f}%"),
                textposition="auto",
                marker=dict(color=top_rows["Reach (%)"], colorscale="Blues", showscale=False),
                hovertemplate="<b>%{customdata}</b><br>Reach: %{x:.2f}%<extra></extra>",
                customdata=top_rows["Combination"],
            ))
            fig.update_layout(
                title=f"Top {top_n} Combinations by Reach  (k={stored_k})",
                xaxis_title="Reach (%)", yaxis_title="",
                height=max(400, top_n * 35), margin=dict(l=10, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("💾 Download Results")
            st.download_button(
                "⬇️ Download Full Results as CSV", data=csv_bytes(res_df),
                file_name=f"turf_analysis_k{stored_k}_results.csv", mime="text/csv",
                width="stretch",
            )

    # ── Tab: Greedy path ──────────────────────────────────────────────────────
    with tab_greedy:
        st.subheader("🪜 Greedy Sequential Portfolio")
        st.markdown("""
        The greedy algorithm builds a portfolio **one item at a time**, always picking the
        item that adds the most new respondents. It is fast and deterministic and shows which
        items contribute the most **incremental reach**, but it is a heuristic: it can miss a
        better combination that the *Top Combinations* search finds.

        The **Incremental Reach** columns show how much *unique, unduplicated* reach each
        touchpoint contributes on top of all previously selected items.
        """)
        if not greedy.portfolio:
            st.info("No items selected by greedy algorithm with current constraints.")
        else:
            rows = [
                {"Step": step, "Item Added": item, "Cumulative Reach (%)": greedy.reach_curve[0],
                 "Incremental Reach (%)": np.nan, "Incremental Reach (Count)": pd.NA,
                 "Note": "✅ must-include"}
                for step, item in enumerate(locked_items, start=1)
            ]
            greedy_items = greedy.portfolio[mi_count:]
            for offset, item in enumerate(greedy_items):
                rows.append({
                    "Step": mi_count + offset + 1, "Item Added": item,
                    "Cumulative Reach (%)": greedy.reach_curve[offset + 1],
                    "Incremental Reach (%)": greedy.marginal_gains[offset],
                    "Incremental Reach (Count)": greedy.marginal_counts[offset],
                    "Note": "",
                })
            greedy_df = pd.DataFrame(rows).astype({"Incremental Reach (Count)": "Int64"})
            st.dataframe(
                greedy_df, width="stretch", hide_index=True,
                column_config={
                    "Cumulative Reach (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Incremental Reach (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Incremental Reach (Count)": st.column_config.NumberColumn(format="%d"),
                },
            )

            rec_size = tc.diminishing_returns_size(
                [mi_count + i + 1 for i in range(len(greedy.marginal_gains))],
                greedy.marginal_gains, DIMINISHING_RETURNS_PCT)
            if rec_size is not None:
                st.info(f"💡 **Recommended portfolio size: {rec_size} items** — marginal reach "
                        f"gain drops below {DIMINISHING_RETURNS_PCT:g}% after step {rec_size}.")
            st.download_button(
                "⬇️ Download Greedy Path as CSV", data=csv_bytes(greedy_df),
                file_name="turf_greedy_path.csv", mime="text/csv",
            )

    # ── Tab: Reach curve ──────────────────────────────────────────────────────
    with tab_curve:
        st.subheader("📈 Reach Curve — How Does Reach Grow with Portfolio Size?")
        st.markdown("""
        This chart shows the **cumulative reach** as items are added via the greedy
        algorithm, and the **marginal gain** per additional item. Use it to identify the
        point of diminishing returns and choose an optimal portfolio size.
        """)
        if len(greedy.reach_curve) < 2:
            st.info("Not enough steps to plot a reach curve.")
        else:
            # x = portfolio size after each step; the first point is the starting set
            # (empty, or the must-include items).
            sizes = [mi_count + i for i in range(len(greedy.reach_curve))]
            labels = ["Must-include" if mi_count else "Baseline", *greedy.portfolio[mi_count:]]

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=sizes, y=greedy.reach_curve, mode="lines+markers+text",
                name="Cumulative Reach (%)",
                text=[f"{r:.1f}%" for r in greedy.reach_curve], textposition="top center",
                line=dict(color="steelblue", width=3), marker=dict(size=9),
            ))
            fig_curve.add_trace(go.Bar(
                x=sizes[1:], y=greedy.marginal_gains, name="Marginal Gain (%)",
                opacity=0.4, marker_color="orange", yaxis="y2",
                hovertemplate="Size %{x}: +%{y:.2f}%<extra></extra>",
            ))
            fig_curve.update_layout(
                xaxis=dict(title="Number of Items in Portfolio (label = item added)",
                           tickmode="array", tickvals=sizes, ticktext=labels, tickangle=-30),
                yaxis=dict(title="Cumulative Reach (%)", range=[0, 105]),
                yaxis2=dict(title="Marginal Gain (%)", overlaying="y", side="right",
                            showgrid=False,
                            range=[0, max(max(greedy.marginal_gains) * 1.3, 10)]),
                legend=dict(x=0.01, y=0.99), height=480,
                title="Reach Curve (Greedy Build-Up)", hovermode="x unified",
            )
            st.plotly_chart(fig_curve, width="stretch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    TURF Analysis Automation Tool · Built with Streamlit
    <br><small>For questions about TURF methodology, consult a market research professional.</small>
</div>
""", unsafe_allow_html=True)
