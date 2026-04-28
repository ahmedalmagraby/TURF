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
        # Cap unique sampling at 75% of the total space to avoid the
        # coupon-collector stall when theoretical_combos ≈ max_combinations.
        safe_max = min(max_combinations, int(theoretical_combos * 0.75))

        rng = np.random.default_rng(42)
        cand_idx_list = list(range(len(candidate_items)))
        sampled = set()
        while len(sampled) < safe_max:
            draw = tuple(sorted(rng.choice(cand_idx_list, extra_k, replace=False).tolist()))
            sampled.add(draw)
        extra_combo_indices = list(sampled)
        is_sampled = True
        coverage_pct = round(safe_max / theoretical_combos * 100, 1)
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

    # When sampling, inject the greedy solution so the "Top Combinations"
    # tab never reports a worse best than the greedy / optimal-by-size tabs.
    if is_sampled:
        g_portfolio, _, _ = greedy_turf(
            df, k, must_include=must_include, must_exclude=must_exclude,
        )
        if len(g_portfolio) >= k:
            greedy_items = g_portfolio[:k]
            greedy_combo = ' + '.join(greedy_items)
            # Only inject if this combination isn't already in the results
            if greedy_combo not in [r['Combination'] for r in results]:
                g_idx = [item_to_idx[it] for it in greedy_items]
                g_reached = np.any(df_arr[:, g_idx], axis=1)
                g_reach_count = int(g_reached.sum())
                g_reach_pct = round(g_reach_count / n_respondents * 100, 2)
                g_frequency = int(df_arr[:, g_idx].sum())
                results.append({
                    'Combination': greedy_combo,
                    'Reach (%)': g_reach_pct,
                    'Reach (Count)': g_reach_count,
                    'Frequency': g_frequency,
                })

    results_df = (pd.DataFrame(results)
                  .sort_values(['Reach (%)', 'Frequency'], ascending=[False, False])
                  .reset_index(drop=True))
    return results_df, is_sampled, theoretical_combos, coverage_pct


# ── Individual reach ──────────────────────────────────────────────────────────
@st.cache_data
def individual_reach(df: pd.DataFrame, must_exclude: tuple = ()):
    """
    Compute standalone reach of every item (excluding must-exclude items).

    Returns
    -------
    reach_df : pd.DataFrame  — columns: Item, Reach (%), Reach (Count)
    """
    n = len(df)
    rows = []
    for col in df.columns:
        if col in must_exclude:
            continue
        count = int((df[col] > 0).sum())
        rows.append({
            'Item': col,
            'Reach (%)': round(count / n * 100, 2),
            'Reach (Count)': count,
        })
    return (pd.DataFrame(rows)
            .sort_values('Reach (%)', ascending=False)
            .reset_index(drop=True))


# ── Optimal combination at each portfolio size ────────────────────────────────
@st.cache_data
def optimal_by_size(df: pd.DataFrame, max_k: int,
                    must_include: tuple = (),
                    must_exclude: tuple = (),
                    max_combinations: int = 15000):
    """
    For each portfolio size 1..max_k, find the best combination by reach.
    Uses exact search when feasible, falls back to the greedy solution otherwise.

    Returns
    -------
    summary_df : pd.DataFrame — one row per portfolio size with:
        Size, Combination, Reach (%), Reach (Count), Incremental Reach (%)
    """
    all_items = df.columns.tolist()
    candidate_items = [it for it in all_items
                       if it not in must_exclude and it not in must_include]
    mi = list(must_include)
    n = len(df)
    arr = df.to_numpy(dtype=np.int8)
    item_idx = {it: i for i, it in enumerate(all_items)}

    # Greedy solution as fallback (always available)
    g_portfolio, g_curve, _ = greedy_turf(df, max_k, must_include, must_exclude)

    rows = []
    prev_reach = 0.0
    for size in range(1, max_k + 1):
        extra_k = size - len(mi)

        # Sizes smaller than must-include count are invalid
        if extra_k < 0:
            continue
        if extra_k > len(candidate_items):
            break

        theo = math.comb(len(candidate_items), extra_k)

        if theo <= max_combinations:
            # Exact search
            res_df, _, _, _ = calculate_turf(
                df, size,
                must_include=must_include,
                must_exclude=must_exclude,
                max_combinations=max_combinations,
            )
            if len(res_df) > 0:
                best = res_df.iloc[0]
                combo = best['Combination']
                reach_pct = best['Reach (%)']
                reach_cnt = best['Reach (Count)']
            else:
                continue
        else:
            # Use greedy approximation for this size
            if size <= len(g_portfolio):
                items_at_size = g_portfolio[:size]
                combo = ' + '.join(items_at_size)
                idxs = [item_idx[it] for it in items_at_size]
                reached = np.any(arr[:, idxs], axis=1).sum()
                reach_pct = round(reached / n * 100, 2)
                reach_cnt = int(reached)
            else:
                continue

        inc = round(reach_pct - prev_reach, 2)
        rows.append({
            'Size': size,
            'Combination': combo,
            'Reach (%)': reach_pct,
            'Reach (Count)': reach_cnt,
            'Incremental Reach (%)': inc,
            'Method': 'Exact' if theo <= max_combinations else 'Greedy approx.',
        })
        prev_reach = reach_pct

    return pd.DataFrame(rows)


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

    # max_k is the *total* portfolio size (including must-include items).
    # The greedy phase only needs to fill the remaining slots.
    greedy_slots = max(0, max_k - len(portfolio))

    for _ in range(min(greedy_slots, len(remaining))):
        best_item, best_gain, best_freq, best_covered = None, -1, -1, covered
        for cand in remaining:
            new_covered = covered | arr[:, item_idx[cand]].astype(bool)
            gain = new_covered.sum() - covered.sum()
            cand_freq = int(arr[:, item_idx[cand]].sum())
            if gain > best_gain or (gain == best_gain and cand_freq > best_freq):
                best_gain, best_item, best_freq, best_covered = gain, cand, cand_freq, new_covered
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
            with st.spinner("Computing individual reach..."):
                ind_reach_df = individual_reach(
                    df, must_exclude=tuple(must_exclude),
                )
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
            with st.spinner("Finding optimal combinations by portfolio size..."):
                opt_size_df = optimal_by_size(
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
                'turf_individual_reach': ind_reach_df,
                'turf_optimal_by_size': opt_size_df,
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
    res_df        = st.session_state.get('turf_results', pd.DataFrame())
    is_sampled    = st.session_state.get('turf_is_sampled', False)
    theo_combos   = st.session_state.get('turf_theo_combos', 0)
    coverage      = st.session_state.get('turf_coverage', 100.0)
    stored_k      = st.session_state.get('turf_k', 0)
    n_resp        = st.session_state.get('turf_n', 0)
    g_portfolio   = st.session_state.get('turf_greedy_portfolio', [])
    g_curve       = st.session_state.get('turf_greedy_curve', [])
    g_marginal    = st.session_state.get('turf_greedy_marginal', [])
    ind_reach_df  = st.session_state.get('turf_individual_reach', pd.DataFrame())
    opt_size_df   = st.session_state.get('turf_optimal_by_size', pd.DataFrame())

    st.header("3️⃣ Analysis Results")

    # Coverage info
    if is_sampled:
        st.warning(f"⚠️ Search space too large ({theo_combos:,} combinations). "
                   f"Evaluated {len(res_df):,} random samples — {coverage}% coverage. "
                   f"Results are a statistical sample; the greedy method below is exact.")
    else:
        st.info(f"ℹ️ Exhaustive search: all {theo_combos:,} combination(s) evaluated (k={stored_k}).")

    tab_individual, tab_optimal_size, tab_full, tab_greedy, tab_curve = st.tabs(
        ["📍 Individual Reach", "🔢 Optimal by Size", "🏆 Top Combinations",
         "🪜 Greedy Path", "📈 Reach Curve"]
    )

    # ── Tab: Individual Reach ─────────────────────────────────────────────────
    with tab_individual:
        st.subheader("📍 Individual Touchpoint Reach")
        st.markdown("""
        Standalone reach of **each individual touchpoint** — the percentage of respondents
        reached by that item alone, independent of any other items.
        """)

        if len(ind_reach_df) == 0:
            st.info("No items to display (all may be excluded).")
        else:
            # Metrics row for top-3
            top3 = ind_reach_df.head(3)
            cols = st.columns(min(3, len(top3)))
            for i, (_, row) in enumerate(top3.iterrows()):
                cols[i].metric(
                    label=row['Item'],
                    value=f"{row['Reach (%)']:.1f}%",
                    help=f"{row['Reach (Count)']} of {n_resp} respondents",
                )

            st.dataframe(
                ind_reach_df,
                use_container_width=True, hide_index=True,
            )

            # Horizontal bar chart
            chart_df = ind_reach_df.iloc[::-1]  # ascending order for horizontal bar
            fig_ind = go.Figure(go.Bar(
                y=chart_df['Item'],
                x=chart_df['Reach (%)'],
                orientation='h',
                text=chart_df['Reach (%)'].map(lambda x: f'{x:.1f}%'),
                textposition='auto',
                marker=dict(
                    color=chart_df['Reach (%)'],
                    colorscale='Teal',
                    showscale=False,
                ),
                hovertemplate='<b>%{y}</b><br>Reach: %{x:.2f}%<extra></extra>',
            ))
            fig_ind.update_layout(
                title='Standalone Reach per Touchpoint',
                xaxis_title='Reach (%)',
                yaxis_title='',
                height=max(350, len(ind_reach_df) * 30),
                margin=dict(l=10, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_ind, use_container_width=True)

            st.download_button(
                label="⬇️ Download Individual Reach as CSV",
                data=ind_reach_df.to_csv(index=False),
                file_name="turf_individual_reach.csv",
                mime="text/csv",
            )

    # ── Tab: Optimal by portfolio size ────────────────────────────────────────
    with tab_optimal_size:
        st.subheader("🔢 Optimal Combinations by Portfolio Size")
        st.markdown("""
        The **best combination** at each portfolio size (1, 2, 3, …) and exactly how much
        **incremental reach** is gained by adding one more touchpoint to the optimal set.
        Use this to decide the ideal portfolio size — where the incremental gain no longer
        justifies the cost of another item.
        """)

        if len(opt_size_df) == 0:
            st.info("No data to display.")
        else:
            st.dataframe(
                opt_size_df,
                use_container_width=True, hide_index=True,
            )

            # Dual-axis chart: reach line + incremental bar
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=opt_size_df['Size'],
                y=opt_size_df['Reach (%)'],
                mode='lines+markers+text',
                name='Reach (%)',
                text=opt_size_df['Reach (%)'].map(lambda x: f'{x:.1f}%'),
                textposition='top center',
                line=dict(color='steelblue', width=3),
                marker=dict(size=9),
            ))
            fig_opt.add_trace(go.Bar(
                x=opt_size_df['Size'],
                y=opt_size_df['Incremental Reach (%)'],
                name='Incremental Reach (%)',
                opacity=0.45,
                marker_color='coral',
                yaxis='y2',
                text=opt_size_df['Incremental Reach (%)'].map(lambda x: f'+{x:.1f}%'),
                textposition='outside',
                hovertemplate='Size %{x}: +%{y:.2f}%<extra></extra>',
            ))
            fig_opt.update_layout(
                xaxis=dict(
                    title='Portfolio Size',
                    tickmode='array',
                    tickvals=opt_size_df['Size'].tolist(),
                ),
                yaxis=dict(title='Cumulative Reach (%)', range=[0, 105]),
                yaxis2=dict(title='Incremental Reach (%)', overlaying='y',
                            side='right', showgrid=False, range=[0, max(opt_size_df['Incremental Reach (%)'].max() * 1.5, 10)]),
                legend=dict(x=0.01, y=0.99),
                height=500,
                title='Optimal Reach by Portfolio Size with Incremental Gains',
                hovermode='x unified',
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            # Highlight diminishing returns
            if len(opt_size_df) > 1:
                inc_vals = opt_size_df['Incremental Reach (%)'].tolist()
                cutoff_idx = next((i for i, v in enumerate(inc_vals[1:], 1) if v < 2.0), None)
                if cutoff_idx is not None:
                    rec_size = int(opt_size_df.iloc[cutoff_idx - 1]['Size'])
                    st.info(f"💡 **Diminishing returns detected:** incremental reach drops below 2% "
                            f"after portfolio size {rec_size}. Consider {rec_size} items as the sweet spot.")

            st.download_button(
                label="⬇️ Download Optimal-by-Size as CSV",
                data=opt_size_df.to_csv(index=False),
                file_name="turf_optimal_by_size.csv",
                mime="text/csv",
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

    # ── Tab: Greedy path ──────────────────────────────────────────────────────
    with tab_greedy:
        st.subheader("🪜 Greedy Sequential Portfolio")
        st.markdown("""
        The greedy algorithm builds a portfolio **one item at a time**, always picking the 
        item that adds the most new respondents. This is exact (no sampling) and shows 
        which items contribute the most **incremental reach**.

        The **Incremental Reach** column shows exactly how much *unique, unduplicated* 
        reach each touchpoint contributes on top of all previously selected items.
        """)

        if not g_portfolio:
            st.info("No items selected by greedy algorithm with current constraints.")
        else:
            # Table: portfolio build-up
            rows = []
            mi_count = len(st.session_state.get('turf_must_include', []))
            for step_i, item in enumerate(g_portfolio):
                locked = "✅ must-include" if step_i < mi_count else ""

                if step_i < mi_count:
                    # Must-include items are added as a bundle before greedy
                    # starts; show the baseline reach (g_curve[0]).
                    reach_after = g_curve[0]
                    rows.append({
                        'Step': step_i + 1,
                        'Item Added': item,
                        'Cumulative Reach (%)': reach_after,
                        'Incremental Reach (%)': "—",
                        'Incremental Reach (Count)': "—",
                        'Note': locked,
                    })
                else:
                    # Greedy items: 0-based index into the greedy portion
                    greedy_idx = step_i - mi_count
                    reach_after = (g_curve[greedy_idx + 1]
                                   if greedy_idx + 1 < len(g_curve)
                                   else g_curve[-1])
                    gain = (g_marginal[greedy_idx]
                            if greedy_idx < len(g_marginal)
                            else 0.0)
                    gain_count = round(gain / 100 * n_resp)
                    rows.append({
                        'Step': step_i + 1,
                        'Item Added': item,
                        'Cumulative Reach (%)': reach_after,
                        'Incremental Reach (%)': gain,
                        'Incremental Reach (Count)': gain_count,
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
