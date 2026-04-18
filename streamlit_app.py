import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import math
import re

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TURF Analysis Tool",
    page_icon="📊",
    layout="wide"
)

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
    - **Marginal Reach**: Additional reach gained by adding one more item
    """)

    st.subheader("Analysis Modes")
    st.markdown("""
    - **Full TURF**: Evaluates every combination of size k to find the optimum
    - **Greedy Sequential**: Builds the portfolio one item at a time, always adding 
      whichever item gains the most new respondents — fast and interpretable
    """)

    st.subheader("Data Format")
    st.markdown("""
    - **Rows**: Each row = one respondent
    - **Columns**: Each column = one item/product/feature
    - **Values**: Binary — 1 (selected) or 0 (not selected)

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

# ── Helper: column-name heuristic ────────────────────────────────────────────
def _is_id_column(col_name: str) -> bool:
    """Word-boundary check to avoid false positives like 'humidity' or 'lipid'."""
    name = col_name.lower().strip()
    patterns = [
        r'\bid\b', r'\bindex\b', r'\brespondent\b', r'\buser\b',
        r'\bcustomer\b', r'\bdemographic\b', r'\btimestamp\b', r'\bdate\b',
    ]
    return any(re.search(p, name) for p in patterns)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_simulated_data(n_respondents: int = 200, n_items: int = 12, seed: int = 42):
    """Generate synthetic binary data with per-item selection probabilities."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.15, 0.45, n_items)          # one probability per item
    data = rng.binomial(1, probs, (n_respondents, n_items))
    columns = [f"Item_{i+1:02d}" for i in range(n_items)]
    return pd.DataFrame(data, columns=columns)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    try:
        df = (pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv')
              else pd.read_excel(uploaded_file))
        return df, None
    except Exception as e:
        return None, f"Error loading file: {e}"

# ── Validation ────────────────────────────────────────────────────────────────
def validate_and_clean_data(df: pd.DataFrame, selected_columns: list):
    df_sel = df[selected_columns].copy()
    numeric_cols = df_sel.select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in selected_columns if c not in numeric_cols]
    if dropped:
        st.warning(f"⚠️ Non-numeric columns excluded: {', '.join(dropped)}")
    if not numeric_cols:
        return None, "No numeric columns found in the selected data."

    df_num = df_sel[numeric_cols].copy()
    nan_count = int(df_num.isna().sum().sum())
    if nan_count:
        st.warning(f"⚠️ {nan_count} missing value(s) found and treated as 0 (not selected).")

    unique_vals = set()
    for col in df_num.columns:
        unique_vals.update(df_num[col].dropna().unique())
    non_binary = [v for v in unique_vals if v not in (0, 1, 0.0, 1.0)]

    if non_binary:
        sample = non_binary[:10]
        suffix = ", ...]" if len(non_binary) > 10 else "]"
        st.warning(f"⚠️ Non-binary values detected: {str(sample)[:-1]}{suffix}. "
                   f"Converting >0 → 1, ≤0 → 0.")
        df_num = (df_num.fillna(0) > 0).astype(np.int8)
    else:
        df_num = df_num.fillna(0).astype(np.int8)

    return df_num, None

# ── Core TURF (pure, no st.* calls) ──────────────────────────────────────────
@st.cache_data
def calculate_turf(df: pd.DataFrame, k: int,
                   must_include: tuple = (),
                   must_exclude: tuple = (),
                   max_combinations: int = 15000):
    """
    Full combinatorial TURF.

    Returns
    -------
    results_df : pd.DataFrame
    is_sampled : bool
    theoretical_combos : int
    coverage_pct : float  — % of search space explored (100 if exhaustive)
    """
    all_items = df.columns.tolist()
    # Apply constraints: remove must-exclude; must-include are locked in
    candidate_items = [it for it in all_items
                       if it not in must_exclude and it not in must_include]
    mi = list(must_include)

    # We need k - len(mi) more items from candidates
    extra_k = k - len(mi)
    if extra_k < 0:
        return pd.DataFrame(), False, 0, 0.0
    if extra_k > len(candidate_items):
        return pd.DataFrame(), False, 0, 0.0

    n_respondents = len(df)
    df_arr = df.to_numpy(dtype=np.int8)
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    # Pre-compute must-include reach mask
    if mi:
        mi_idx = [item_to_idx[it] for it in mi]
        base_reached = np.any(df_arr[:, mi_idx], axis=1)
    else:
        base_reached = np.zeros(n_respondents, dtype=bool)

    theoretical_combos = math.comb(len(candidate_items), extra_k)

    if theoretical_combos > max_combinations:
        rng = np.random.default_rng(42)
        cand_idx_list = list(range(len(candidate_items)))
        sampled = set()
        while len(sampled) < max_combinations:
            draw = tuple(sorted(rng.choice(cand_idx_list, extra_k, replace=False).tolist()))
            sampled.add(draw)
        extra_combo_indices = list(sampled)
        is_sampled = True
        coverage_pct = round(max_combinations / theoretical_combos * 100, 1)
    else:
        extra_combo_indices = list(combinations(range(len(candidate_items)), extra_k))
        is_sampled = False
        coverage_pct = 100.0

    cand_arr = df_arr[:, [item_to_idx[it] for it in candidate_items]]

    results = []
    for extra_idx in extra_combo_indices:
        extra_names = [candidate_items[i] for i in extra_idx]
        all_names = mi + extra_names

        extra_cols = cand_arr[:, list(extra_idx)]
        reached = base_reached | np.any(extra_cols, axis=1)
        reach_count = int(reached.sum())
        reach_pct = reach_count / n_respondents * 100

        # Frequency = total selections from this full combo
        full_idx = [item_to_idx[n] for n in all_names]
        frequency = int(df_arr[:, full_idx].sum())

        results.append({
            'Combination': ' + '.join(all_names),
            'Reach (%)': round(reach_pct, 2),
            'Reach (Count)': reach_count,
            'Frequency': frequency,
        })

    results_df = (pd.DataFrame(results)
                  .sort_values(['Reach (%)', 'Frequency'], ascending=[False, False])
                  .reset_index(drop=True))
    return results_df, is_sampled, theoretical_combos, coverage_pct


# ── Greedy sequential TURF ────────────────────────────────────────────────────
@st.cache_data
def greedy_turf(df: pd.DataFrame, max_k: int,
                must_include: tuple = (),
                must_exclude: tuple = ()):
    """
    Build a portfolio greedily: at each step add the item that gains the most reach.

    Returns
    -------
    portfolio : list[str]            — ordered list of selected items
    reach_curve : list[float]        — reach (%) after each item is added
    marginal_gains : list[float]     — incremental reach gain at each step
    """
    all_items = df.columns.tolist()
    remaining = [it for it in all_items if it not in must_exclude and it not in must_include]
    n = len(df)
    arr = df.to_numpy(dtype=np.int8)
    item_idx = {it: i for i, it in enumerate(all_items)}

    # Start from must-include set
    portfolio = list(must_include)
    if portfolio:
        covered = np.any(arr[:, [item_idx[it] for it in portfolio]], axis=1)
    else:
        covered = np.zeros(n, dtype=bool)

    reach_curve = [round(covered.sum() / n * 100, 2)]  # baseline
    marginal_gains = []

    for _ in range(min(max_k, len(remaining))):
        best_item, best_gain, best_covered = None, -1, covered
        for cand in remaining:
            new_covered = covered | arr[:, item_idx[cand]].astype(bool)
            gain = new_covered.sum() - covered.sum()
            if gain > best_gain:
                best_gain, best_item, best_covered = gain, cand, new_covered
        if best_item is None:
            break
        portfolio.append(best_item)
        remaining.remove(best_item)
        covered = best_covered
        reach_curve.append(round(covered.sum() / n * 100, 2))
        marginal_gains.append(round(best_gain / n * 100, 2))

    return portfolio, reach_curve, marginal_gains


# ══════════════════════════════════════════════════════════════════════════════
#  UI — Data source
# ══════════════════════════════════════════════════════════════════════════════
st.header("1️⃣ Data Source Selection")

data_source = st.radio(
    "Choose your data source:",
    options=["Use Simulated Data", "Upload Own Data"],
    help="Select simulated data to test the tool, or upload your own CSV/Excel file."
)

df = None
df_raw = None

if data_source == "Use Simulated Data":
    st.info("📊 Using simulated data for demonstration purposes.")
    col1, col2 = st.columns(2)
    with col1:
        n_respondents = st.slider("Number of Respondents", 50, 500, 200, 50)
    with col2:
        n_items = st.slider("Number of Items", 5, 20, 12, 1)
    df = generate_simulated_data(n_respondents, n_items)
    st.success(f"✅ Generated {n_respondents} respondents × {n_items} items")

else:
    st.info("📁 Upload your binary data file (CSV or Excel)")
    uploaded_file = st.file_uploader(
        "Choose a file", type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with binary data (0s and 1s)"
    )
    if uploaded_file is not None:
        df_raw, error = load_data(uploaded_file)
        if error:
            st.error(error)
        else:
            st.success(f"✅ File loaded: {len(df_raw)} rows × {len(df_raw.columns)} columns")
            st.subheader("📋 Select Columns to Analyze")
            st.markdown("""
            **Important:** Only select columns that represent items/products/features.
            ⚠️ Exclude respondent IDs, demographics, timestamps.
            """)
            all_columns = df_raw.columns.tolist()
            potential_id_cols = [c for c in all_columns if _is_id_column(c)]
            default_cols = [c for c in all_columns if c not in potential_id_cols]
            if potential_id_cols:
                st.warning(f"⚠️ Likely ID/metadata columns excluded by default: {', '.join(potential_id_cols)}")
            selected_columns = st.multiselect(
                "Select columns to include in TURF analysis:",
                options=all_columns, default=default_cols,
                help="Choose only item/product columns."
            )
            if len(selected_columns) < 2:
                st.error("❌ Please select at least 2 columns for analysis.")
            else:
                df, error = validate_and_clean_data(df_raw, selected_columns)
                if error:
                    st.error(error)
                else:
                    st.success(f"✅ Ready: {len(df)} respondents × {len(df.columns)} items")

# ── Data preview ───────────────────────────────────────────────────────────────
if df is not None:
    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Respondents", len(df))
        col2.metric("Items", len(df.columns))
        total_cells = len(df) * len(df.columns)
        col3.metric("Overall Selection Rate",
                    f"{df.to_numpy().sum() / total_cells * 100:.1f}%")
        st.subheader("Item Selection Rates")
        item_rates = (df.sum() / len(df) * 100).sort_values(ascending=False)
        fig_prev = px.bar(x=item_rates.index, y=item_rates.values,
                          labels={'x': 'Item', 'y': 'Selection Rate (%)'},
                          title='Individual Item Selection Rates')
        st.plotly_chart(fig_prev, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — Analysis configuration
# ══════════════════════════════════════════════════════════════════════════════
if df is not None:
    st.header("2️⃣ Analysis Configuration")

    all_items = df.columns.tolist()

    # ── Constraints ──────────────────────────────────────────────────────────
    with st.expander("🔒 Item Constraints (Optional)", expanded=False):
        st.markdown("""
        **Must-Include**: items that are always in every portfolio (e.g., a flagship product).  
        **Must-Exclude**: items that should never appear (e.g., discontinued lines).
        """)
        col_mi, col_me = st.columns(2)
        with col_mi:
            must_include = st.multiselect(
                "✅ Must-Include items:",
                options=all_items, default=[],
                help="These items are locked into every combination."
            )
        with col_me:
            excludable = [it for it in all_items if it not in must_include]
            must_exclude = st.multiselect(
                "🚫 Must-Exclude items:",
                options=excludable, default=[],
                help="These items will never appear in any combination."
            )

    # Validate constraints
    candidate_items = [it for it in all_items if it not in must_exclude and it not in must_include]
    n_mi = len(must_include)
    n_cand = len(candidate_items)
    constraint_ok = True

    if n_mi + n_cand < 2:
        st.error("❌ Too many constraints — not enough items remain for analysis.")
        constraint_ok = False

    # ── k slider ─────────────────────────────────────────────────────────────
    if constraint_ok:
        col1, col2 = st.columns([2, 1])
        with col1:
            max_k = max(2, min(len(all_items), 12))
            min_k = max(2, n_mi + 1) if n_cand >= 1 else 2
            min_k = min(min_k, max_k)
            k = st.slider(
                "Portfolio Size (k)",
                min_value=min_k, max_value=max_k,
                value=max(min_k, min(3, max_k)),
                help="Total number of items in each combination (includes must-include items)"
            )
        with col2:
            extra_k = k - n_mi
            if extra_k >= 0 and n_cand >= extra_k:
                theo = math.comb(n_cand, extra_k)
                st.metric("Search Space", f"{theo:,} combos")
            else:
                st.metric("Search Space", "N/A")
                constraint_ok = False

    # ── Run button ────────────────────────────────────────────────────────────
    if constraint_ok:
        # Build a fingerprint so we can detect stale results
        import hashlib, json
        cfg = json.dumps({
            'cols': all_items, 'k': k, 'n': len(df),
            'mi': sorted(must_include), 'me': sorted(must_exclude)
        }, sort_keys=True)
        cfg_hash = hashlib.md5(cfg.encode()).hexdigest()

        if st.button("🚀 Run TURF Analysis", type="primary", use_container_width=True):
            with st.spinner("Running full TURF analysis..."):
                res_df, is_sampled, theo_combos, coverage = calculate_turf(
                    df, k,
                    must_include=tuple(must_include),
                    must_exclude=tuple(must_exclude),
                )
            with st.spinner("Running greedy sequential analysis..."):
                g_portfolio, g_reach_curve, g_marginal = greedy_turf(
                    df, max_k,
                    must_include=tuple(must_include),
                    must_exclude=tuple(must_exclude),
                )
            st.session_state.update({
                'turf_results': res_df,
                'turf_is_sampled': is_sampled,
                'turf_theo_combos': theo_combos,
                'turf_coverage': coverage,
                'turf_k': k,
                'turf_n': len(df),
                'turf_greedy_portfolio': g_portfolio,
                'turf_greedy_curve': g_reach_curve,
                'turf_greedy_marginal': g_marginal,
                'turf_cfg_hash': cfg_hash,
                'turf_must_include': must_include,
            })

        # ── Stale results banner ──────────────────────────────────────────────
        if ('turf_results' in st.session_state and
                st.session_state.get('turf_cfg_hash') != cfg_hash):
            st.warning("⚠️ The configuration has changed since the last analysis. "
                       "Click **Run TURF Analysis** to refresh results.")

# ══════════════════════════════════════════════════════════════════════════════
#  UI — Results
# ══════════════════════════════════════════════════════════════════════════════
if 'turf_results' in st.session_state:
    res_df        = st.session_state['turf_results']
    is_sampled    = st.session_state['turf_is_sampled']
    theo_combos   = st.session_state['turf_theo_combos']
    coverage      = st.session_state['turf_coverage']
    stored_k      = st.session_state['turf_k']
    n_resp        = st.session_state['turf_n']
    g_portfolio   = st.session_state['turf_greedy_portfolio']
    g_curve       = st.session_state['turf_greedy_curve']
    g_marginal    = st.session_state['turf_greedy_marginal']

    st.header("3️⃣ Analysis Results")

    # Coverage info
    if is_sampled:
        st.warning(f"⚠️ Search space too large ({theo_combos:,} combinations). "
                   f"Evaluated {len(res_df):,} random samples — {coverage}% coverage. "
                   f"Results are a statistical sample; the greedy method below is exact.")
    else:
        st.info(f"ℹ️ Exhaustive search: all {theo_combos:,} combination(s) evaluated (k={stored_k}).")

    tab_full, tab_greedy, tab_curve = st.tabs(
        ["🏆 Top Combinations", "🪜 Greedy Path", "📈 Reach Curve"]
    )

    # ── Tab 1: Full TURF results ──────────────────────────────────────────────
    with tab_full:
        if len(res_df) == 0:
            st.error("No valid combinations found with current constraints.")
        else:
            top = res_df.iloc[0]
            not_reached = 100 - top['Reach (%)']
            st.success("✅ Analysis Complete!")

            st.subheader("🏆 Best Portfolio")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Reach", f"{top['Reach (%)']:.1f}%")
            m2.metric("Respondents Reached", f"{top['Reach (Count)']:,} / {n_resp:,}")
            m3.metric("Not Reached", f"{not_reached:.1f}%")
            m4.metric("Total Selections", f"{top['Frequency']:,}")
            st.markdown(f"**Portfolio:** `{top['Combination']}`")

            st.subheader("📊 Top 20 Combinations")
            st.dataframe(
                res_df.head(20)[['Combination', 'Reach (%)', 'Reach (Count)', 'Frequency']],
                use_container_width=True, hide_index=True
            )

            # Chart
            top_n = min(15, len(res_df))
            top_rows = res_df.head(top_n).copy()
            top_rows['Label'] = top_rows['Combination'].apply(
                lambda s: s if len(s) <= 45 else s[:45] + '…')
            top_rows = top_rows.iloc[::-1]  # ascending for horizontal bar

            fig = go.Figure(go.Bar(
                y=top_rows['Label'],
                x=top_rows['Reach (%)'],
                orientation='h',
                text=top_rows['Reach (%)'].map(lambda x: f'{x:.1f}%'),
                textposition='auto',
                marker=dict(
                    color=top_rows['Reach (%)'],
                    colorscale='Blues',
                    showscale=False,
                ),
                hovertemplate='<b>%{customdata}</b><br>Reach: %{x:.2f}%<extra></extra>',
                customdata=top_rows['Combination'],
            ))
            fig.update_layout(
                title=f'Top {top_n} Combinations by Reach  (k={stored_k})',
                xaxis_title='Reach (%)',
                yaxis_title='',
                height=max(400, top_n * 35),
                margin=dict(l=10, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download
            st.subheader("💾 Download Results")
            st.download_button(
                label="⬇️ Download Full Results as CSV",
                data=res_df.to_csv(index=False),
                file_name=f"turf_analysis_k{stored_k}_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ── Tab 2: Greedy path ────────────────────────────────────────────────────
    with tab_greedy:
        st.subheader("🪜 Greedy Sequential Portfolio")
        st.markdown("""
        The greedy algorithm builds a portfolio **one item at a time**, always picking the 
        item that adds the most new respondents. This is exact (no sampling) and shows 
        which items contribute the most incremental reach.
        """)

        if not g_portfolio:
            st.info("No items selected by greedy algorithm with current constraints.")
        else:
            # Table: portfolio build-up
            rows = []
            mi_count = len(st.session_state.get('turf_must_include', []))
            for step_i, item in enumerate(g_portfolio):
                reach_after = g_curve[step_i + 1] if step_i + 1 < len(g_curve) else g_curve[-1]
                gain = g_marginal[step_i] if step_i < len(g_marginal) else 0.0
                locked = "✅ must-include" if step_i < mi_count else ""
                rows.append({
                    'Step': step_i + 1,
                    'Item Added': item,
                    'Cumulative Reach (%)': reach_after,
                    'Marginal Gain (%)': gain if step_i >= mi_count else "—",
                    'Note': locked,
                })
            greedy_df = pd.DataFrame(rows)
            st.dataframe(greedy_df, use_container_width=True, hide_index=True)

            # Highlight the recommended cutoff (where marginal gain < 2%)
            effective_gains = [g for g in g_marginal if isinstance(g, float)]
            if effective_gains:
                cutoff = next((i for i, g in enumerate(effective_gains) if g < 2.0),
                              len(effective_gains))
                if cutoff < len(g_portfolio) - mi_count:
                    st.info(f"💡 **Recommended portfolio size: {mi_count + cutoff} items** — "
                            f"marginal reach gain drops below 2% after step {mi_count + cutoff}.")

            st.download_button(
                label="⬇️ Download Greedy Path as CSV",
                data=greedy_df.to_csv(index=False),
                file_name="turf_greedy_path.csv",
                mime="text/csv",
            )

    # ── Tab 3: Reach curve ────────────────────────────────────────────────────
    with tab_curve:
        st.subheader("📈 Reach Curve — How Does Reach Grow with Portfolio Size?")
        st.markdown("""
        This chart shows the **cumulative reach** as items are added via the greedy 
        algorithm, and the **marginal gain** per additional item. Use it to identify the 
        point of diminishing returns and choose an optimal portfolio size.
        """)

        if len(g_curve) < 2:
            st.info("Not enough steps to plot a reach curve.")
        else:
            steps = list(range(len(g_curve)))
            labels = ['Baseline'] + g_portfolio[:len(steps) - 1]

            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=steps, y=g_curve,
                mode='lines+markers+text',
                name='Cumulative Reach (%)',
                text=[f'{r:.1f}%' for r in g_curve],
                textposition='top center',
                line=dict(color='steelblue', width=3),
                marker=dict(size=9),
            ))

            if g_marginal:
                mi_count = len(st.session_state.get('turf_must_include', []))
                marginal_x = list(range(mi_count + 1, len(g_curve)))
                marginal_y = g_marginal[:]
                # pad if lengths differ
                marginal_y = marginal_y[:len(marginal_x)]
                fig_curve.add_trace(go.Bar(
                    x=marginal_x,
                    y=marginal_y,
                    name='Marginal Gain (%)',
                    opacity=0.4,
                    marker_color='orange',
                    yaxis='y2',
                    hovertemplate='Step %{x}: +%{y:.2f}%<extra></extra>',
                ))

            fig_curve.update_layout(
                xaxis=dict(
                    title='Number of Items in Portfolio',
                    tickmode='array',
                    tickvals=steps,
                    ticktext=labels,
                    tickangle=-30,
                ),
                yaxis=dict(title='Cumulative Reach (%)', range=[0, 105]),
                yaxis2=dict(title='Marginal Gain (%)', overlaying='y',
                            side='right', showgrid=False, range=[0, 30]),
                legend=dict(x=0.01, y=0.99),
                height=480,
                title='Reach Curve (Greedy Build-Up)',
                hovermode='x unified',
            )
            st.plotly_chart(fig_curve, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    TURF Analysis Automation Tool · Built with Streamlit
    <br><small>For questions about TURF methodology, consult a market research professional.</small>
</div>
""", unsafe_allow_html=True)
