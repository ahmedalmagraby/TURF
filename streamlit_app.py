import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
import math
import io

# Page configuration
st.set_page_config(
    page_title="TURF Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 TURF Analysis Automation Tool")
st.markdown("""
**Total Unduplicated Reach and Frequency (TURF) Analysis** helps identify the optimal 
combination of items that maximizes audience reach while minimizing overlap.
""")

# Sidebar - Methodology & Guide
with st.sidebar:
    st.header("📖 Methodology & Guide")
    
    st.subheader("What is TURF?")
    st.markdown("""
    TURF analysis identifies the best combination of items (products, features, messages) 
    that reaches the maximum number of unique respondents.
    """)
    
    st.subheader("Key Metrics")
    st.markdown("""
    - **Reach (%)**: Percentage of unique respondents who selected at least one item 
      in the combination (Logical OR)
    - **Frequency**: Total count of selections across all items in the combination
    """)
    
    st.subheader("Data Format Requirements")
    st.markdown("""
    For uploaded files:
    - **Rows**: Each row represents one respondent
    - **Columns**: Each column represents one item/product/feature
    - **Values**: Binary (0 or 1)
        - 1 = Respondent selected/preferred this item
        - 0 = Respondent did not select this item
    
    **Example:**
    ```
    Item_A  Item_B  Item_C
    1       0       1
    0       1       1
    1       1       0
    ```
    
    ⚠️ **Important**: Exclude ID columns and demographic variables!
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. Choose your data source (Upload or Simulate)
    2. If uploading, select which columns to analyze
    3. Select the maximum portfolio size (k)
    4. Click "Run TURF Analysis"
    5. Review the top combinations
    6. Interpret the reach and frequency metrics
    """)

# Cache data generation
@st.cache_data
def generate_simulated_data(n_respondents=200, n_items=12, seed=42):
    """Generate synthetic binary data for testing."""
    np.random.seed(seed)
    
    # Generate realistic binary data with varying probabilities
    data = np.random.binomial(1, np.random.uniform(0.15, 0.45, n_items), 
                              (n_respondents, n_items))
    
    # Create column names
    columns = [f"Item_{chr(65+i)}" for i in range(n_items)]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Cache data loading
@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # Excel
            df = pd.read_excel(uploaded_file)
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def validate_and_clean_data(df, selected_columns):
    """Validate and clean the data to ensure it's binary."""
    # Use only selected columns
    df_selected = df[selected_columns].copy()
    
    # Check if data is numeric
    numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None, "No numeric columns found in the selected data."
    
    # Use only numeric columns
    df_numeric = df_selected[numeric_cols].copy()
    
    # Check for non-binary values
    unique_values = set()
    for col in df_numeric.columns:
        unique_values.update(df_numeric[col].dropna().unique())
    
    non_binary = [v for v in unique_values if v not in [0, 1, 0.0, 1.0]]
    
    if non_binary:
        st.warning(f"⚠️ Non-binary values detected: {non_binary}. Converting values >0 to 1, and values ≤0 to 0.")
        # Coerce to binary
        df_numeric = (df_numeric > 0).astype(int)
    
    # Ensure all values are 0 or 1
    df_numeric = df_numeric.fillna(0).astype(int)
    
    return df_numeric, None

# Cache TURF calculation
@st.cache_data
def calculate_turf(df, k, max_combinations=10000):
    """
    Calculate TURF metrics for all combinations of size k.
    
    Parameters:
    - df: Binary DataFrame (respondents x items)
    - k: Portfolio size (number of items in combination)
    - max_combinations: Maximum number of combinations to evaluate
    
    Returns:
    - DataFrame with columns: Combination, Reach, Frequency
    """
    items = df.columns.tolist()
    n_items = len(items)
    n_respondents = len(df)
    
    # Calculate theoretical number of combinations using math.comb
    theoretical_combos = math.comb(n_items, k)
    
    results = []
    
    # Check if we need to sample
    if theoretical_combos > max_combinations:
        st.warning(f"⚠️ Total possible combinations: {theoretical_combos:,}. Evaluating a random sample of {max_combinations:,}.")
        
        # Generate random sample of combinations without creating full list
        np.random.seed(42)
        sampled_combos = set()
        
        # Keep sampling until we have enough unique combinations
        while len(sampled_combos) < max_combinations:
            # Randomly select k items
            random_indices = np.random.choice(n_items, k, replace=False)
            combo = tuple(sorted([items[i] for i in random_indices]))
            sampled_combos.add(combo)
        
        all_combos = list(sampled_combos)
        is_sampled = True
    else:
        # Generate all combinations
        all_combos = list(combinations(items, k))
        is_sampled = False
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combos = len(all_combos)
    
    for idx, combo in enumerate(all_combos):
        # Update progress every 100 iterations or at the end
        if idx % 100 == 0 or idx == total_combos - 1:
            progress = (idx + 1) / total_combos
            progress_bar.progress(progress)
            status_text.text(f"Analyzing combination {idx + 1:,} of {total_combos:,}...")
        
        # Get subset of data for this combination
        subset = df[list(combo)]
        
        # Reach: Number of respondents with at least one 1 (Logical OR)
        reach_count = (subset.sum(axis=1) > 0).sum()
        reach_pct = (reach_count / n_respondents) * 100
        
        # Frequency: Total sum of 1s across all items
        frequency = subset.sum().sum()
        
        results.append({
            'Combination': ' + '.join(combo),
            'Items': combo,
            'Reach (%)': round(reach_pct, 2),
            'Reach (Count)': reach_count,
            'Frequency': frequency
        })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Add info about sampling
    if is_sampled:
        st.info(f"ℹ️ Analyzed {max_combinations:,} random combinations out of {theoretical_combos:,} possible. Results represent a statistical sample.")
    else:
        st.info(f"ℹ️ Analyzed all {theoretical_combos:,} possible combinations.")
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=['Reach (%)', 'Frequency'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    return results_df

# Main app logic
st.header("1️⃣ Data Source Selection")

# Data source selection
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
    
else:  # Upload Own Data
    st.info("📁 Upload your binary data file (CSV or Excel)")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with binary data (0s and 1s)"
    )
    
    if uploaded_file is not None:
        df_raw, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.success(f"✅ File loaded successfully: {len(df_raw)} rows × {len(df_raw.columns)} columns")
            
            # Column selection
            st.subheader("📋 Select Columns to Analyze")
            st.markdown("""
            **Important:** Only select the columns that represent items/products/features to analyze.
            
            ⚠️ **Exclude:** Respondent IDs, demographic variables, timestamps, or any non-item columns.
            """)
            
            all_columns = df_raw.columns.tolist()
            
            # Try to detect potential ID columns
            potential_id_cols = [col for col in all_columns if 
                               any(keyword in col.lower() for keyword in ['id', 'respondent', 'user', 'customer', 'index'])]
            
            if potential_id_cols:
                st.warning(f"⚠️ Potential ID columns detected: {', '.join(potential_id_cols)}. Consider excluding these.")
            
            # Default to all columns selected
            selected_columns = st.multiselect(
                "Select columns to include in TURF analysis:",
                options=all_columns,
                default=all_columns,
                help="Choose only the item/product columns. Uncheck ID columns and demographics."
            )
            
            if len(selected_columns) < 2:
                st.error("❌ Please select at least 2 columns for analysis.")
            else:
                df, error = validate_and_clean_data(df_raw, selected_columns)
                
                if error:
                    st.error(error)
                else:
                    st.success(f"✅ Ready to analyze: {len(df)} respondents × {len(df.columns)} items")

# Show data preview
if df is not None:
    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Respondents", len(df))
        with col2:
            st.metric("Total Items", len(df.columns))
        with col3:
            st.metric("Overall Selection Rate", f"{(df.sum().sum() / (len(df) * len(df.columns)) * 100):.1f}%")
        
        # Item popularity
        st.subheader("Item Selection Rates")
        item_rates = (df.sum() / len(df) * 100).sort_values(ascending=False)
        fig = px.bar(
            x=item_rates.index,
            y=item_rates.values,
            labels={'x': 'Item', 'y': 'Selection Rate (%)'},
            title='Individual Item Selection Rates'
        )
        st.plotly_chart(fig, use_container_width=True)

# Analysis section
if df is not None:
    st.header("2️⃣ Analysis Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        max_k = min(len(df.columns), 10)
        k = st.slider(
            "Maximum Portfolio Size (k)",
            min_value=2,
            max_value=max_k,
            value=min(3, max_k),
            help="Select how many items to include in each combination"
        )
    
    with col2:
        theoretical_combos = math.comb(len(df.columns), k)
        st.metric("Possible Combinations", f"{theoretical_combos:,}")
    
    # Run analysis button
    if st.button("🚀 Run TURF Analysis", type="primary", use_container_width=True):
        st.header("3️⃣ Analysis Results")
        
        with st.spinner("Calculating TURF metrics..."):
            results_df = calculate_turf(df, k)
        
        # Top result interpretation
        if len(results_df) > 0:
            top_result = results_df.iloc[0]
            reach_pct = top_result['Reach (%)']
            not_reached_pct = 100 - reach_pct
            
            st.success("✅ Analysis Complete!")
            
            st.subheader("🏆 Top Combination")
            st.markdown(f"""
            **Best Portfolio:** {top_result['Combination']}
            
            **Interpretation:**
            - This combination reaches **{reach_pct}%** of your audience
            - That means only **{not_reached_pct:.1f}%** of respondents selected none of these items
            - Total selections across these items: **{top_result['Frequency']}**
            - Number of unique respondents reached: **{top_result['Reach (Count)']} out of {len(df)}**
            """)
            
            # Display top 20 results
            st.subheader("📊 Top 20 Combinations")
            display_df = results_df.head(20)[['Combination', 'Reach (%)', 'Reach (Count)', 'Frequency']]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=False
            )
            
            # Visualization
            st.subheader("📈 Reach Comparison (Top 15)")
            top_15 = results_df.head(15)
            
            # Truncate long combination names for readability
            def truncate_name(name, max_length=40):
                if len(name) > max_length:
                    return name[:max_length] + "..."
                return name
            
            truncated_names = [truncate_name(name) for name in top_15['Combination']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=truncated_names,
                x=top_15['Reach (%)'],
                orientation='h',
                text=top_15['Reach (%)'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto',
                marker_color='steelblue',
                hovertemplate='<b>%{customdata}</b><br>' +
                             'Reach: %{x:.2f}%<br>' +
                             '<extra></extra>',
                customdata=top_15['Combination']  # Full name in hover
            ))
            
            fig.update_layout(
                title=f'Top 15 Combinations by Reach (Portfolio Size: {k})',
                xaxis_title='Reach (%)',
                yaxis_title='Combination',
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results as CSV",
                data=csv,
                file_name=f"turf_analysis_k{k}_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>TURF Analysis Automation Tool | Built with Streamlit</p>
    <p><small>For questions about TURF methodology, consult a market research professional.</small></p>
</div>
""", unsafe_allow_html=True)
