import os
import sys
import io
import tempfile
import traceback
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# Add current directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import the fixed runner
try:
    from fixed_runner import run_inference_csv_fixed, find_checkpoints_fixed
    _runner_import_error = None
except Exception as e:
    _runner_import_error = e

st.set_page_config(page_title="Model vs Baseline Comparison App", layout="wide")
st.title("Model vs Baseline Comparison App")
st.caption("Compare pre-trained model predictions against baseline methods and ground truth on your unseen data.")

if _runner_import_error:
    st.error(f"Could not import fixed runner: {_runner_import_error}")
    st.info("""
    **To fix this:**
    1. Make sure `fixed_runner.py` is in the same directory as `app.py`
    2. Run training first: `python main.py --epochs 3 --use_gpu`
    3. Check that all required packages are installed
    """)
    st.stop()

# ---------------------- Session Defaults ----------------------
if "ckpt_root" not in st.session_state:
    st.session_state.ckpt_root = "./demo_outputs"
if "selected_ckpt_label" not in st.session_state:
    st.session_state.selected_ckpt_label = None
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# ---------------------- Enhanced Helpers ----------------------
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()

def is_classification_like(y: np.ndarray, max_classes: int = 20, tol: float = 1e-6) -> bool:
    if y.size == 0:
        return False
    rounded = np.round(y)
    if np.max(np.abs(y - rounded)) > tol:
        return False
    unique_vals = np.unique(rounded)
    return unique_vals.size <= max_classes

def generate_baseline_predictions(y_true: np.ndarray, method: str = "last_value") -> np.ndarray:
    """Generate baseline predictions using simple methods."""
    if method == "last_value":
        if y_true.ndim == 1:
            return np.full_like(y_true, y_true[-1] if len(y_true) > 0 else 0)
        else:
            return np.tile(y_true[-1, :], (y_true.shape[0], 1))
   
    elif method == "mean":
        if y_true.ndim == 1:
            return np.full_like(y_true, np.mean(y_true))
        else:
            means = np.mean(y_true, axis=0)
            return np.tile(means, (y_true.shape[0], 1))
   
    elif method == "zero":
        return np.zeros_like(y_true)
   
    elif method == "random":
        if y_true.ndim == 1:
            return np.random.normal(np.mean(y_true), np.std(y_true), size=y_true.shape)
        else:
            means = np.mean(y_true, axis=0)
            stds = np.std(y_true, axis=0)
            return np.random.normal(means, stds, size=y_true.shape)
   
    else:
        return np.zeros_like(y_true)

def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_name: str, model_type: str) -> dict:
    """Compute both regression and classification metrics for comprehensive comparison"""
    metrics = {
        "label": label_name,
        "model_type": model_type,
        "problem_type": "unknown"
    }
   
    # Regression metrics
    try:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    except:
        metrics["mae"] = metrics["rmse"] = metrics["r2"] = metrics["mse"] = np.nan
   
    # Classification metrics (if applicable)
    if is_classification_like(y_true):
        yt = np.round(y_true).astype(int)
        yp = np.round(y_pred).astype(int)
        try:
            metrics["accuracy"] = float(accuracy_score(yt, yp))
            metrics["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
            metrics["problem_type"] = "classification"
        except:
            metrics["accuracy"] = metrics["f1_macro"] = np.nan
    else:
        metrics["problem_type"] = "regression"
   
    return metrics

def render_comparison_metrics(metrics_baseline: dict, metrics_model: dict):
    """Render side-by-side comparison of baseline vs model metrics"""
    col1, col2, col3 = st.columns([1, 1, 1])
   
    with col1:
        st.subheader("  Baseline")
        st.metric("MAE", f"{metrics_baseline['mae']:.4f}")
        st.metric("RMSE", f"{metrics_baseline['rmse']:.4f}")
        st.metric("R²", f"{metrics_baseline['r2']:.4f}")
        if metrics_baseline['problem_type'] == "classification":
            st.metric("Accuracy", f"{metrics_baseline['accuracy']:.4f}")
            st.metric("F1 Macro", f"{metrics_baseline['f1_macro']:.4f}")
   
    with col2:
        st.subheader(" Model")
        st.metric("MAE", f"{metrics_model['mae']:.4f}")
        st.metric("RMSE", f"{metrics_model['rmse']:.4f}")
        st.metric("R²", f"{metrics_model['r2']:.4f}")
        if metrics_model['problem_type'] == "classification":
            st.metric("Accuracy", f"{metrics_model['accuracy']:.4f}")
            st.metric("F1 Macro", f"{metrics_model['f1_macro']:.4f}")
   
    with col3:
        st.subheader(" Improvement")
        # Handle division by zero
        mae_improvement = 0
        if metrics_baseline['mae'] != 0:
            mae_improvement = (metrics_baseline['mae'] - metrics_model['mae']) / metrics_baseline['mae'] * 100
       
        rmse_improvement = 0
        if metrics_baseline['rmse'] != 0:
            rmse_improvement = (metrics_baseline['rmse'] - metrics_model['rmse']) / metrics_baseline['rmse'] * 100
       
        r2_improvement = (metrics_model['r2'] - metrics_baseline['r2']) * 100
       
        st.metric("MAE Improvement", f"{mae_improvement:+.1f}%",
                 delta=f"{mae_improvement:+.1f}%" if mae_improvement > 0 else None)
        st.metric("RMSE Improvement", f"{rmse_improvement:+.1f}%",
                 delta=f"{rmse_improvement:+.1f}%" if rmse_improvement > 0 else None)
        st.metric("R² Improvement", f"{r2_improvement:+.1f}%",
                 delta=f"{r2_improvement:+.1f}%" if r2_improvement > 0 else None)
       
        if metrics_model['problem_type'] == "classification":
            acc_improvement = (metrics_model['accuracy'] - metrics_baseline['accuracy']) * 100
            f1_improvement = (metrics_model['f1_macro'] - metrics_baseline['f1_macro']) * 100
            st.metric("Accuracy Improvement", f"{acc_improvement:+.1f}%",
                     delta=f"{acc_improvement:+.1f}%" if acc_improvement > 0 else None)
            st.metric("F1 Improvement", f"{f1_improvement:+.1f}%",
                     delta=f"{f1_improvement:+.1f}%" if f1_improvement > 0 else None)

def plot_comparison_timeseries(y_true: np.ndarray, y_pred_baseline: np.ndarray, y_pred_model: np.ndarray, label_name: str):
    """Plot comparison between true values, baseline predictions, and model predictions"""
    fig, ax = plt.subplots(figsize=(10, 4))
   
    ax.plot(y_true, label='True Values', linewidth=2, color='black', alpha=0.8)
    ax.plot(y_pred_baseline, label='Baseline Predictions', linewidth=1.5, color='red', alpha=0.7, linestyle='--')
    ax.plot(y_pred_model, label='Model Predictions', linewidth=1.5, color='blue', alpha=0.7)
   
    ax.set_title(f"Time Series Comparison — {label_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_comparison_scatter(y_true: np.ndarray, y_pred_baseline: np.ndarray, y_pred_model: np.ndarray, label_name: str):
    """Plot scatter comparison between baseline and model predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
    # Baseline scatter
    ax1.scatter(y_true, y_pred_baseline, alpha=0.6, s=30, color='red')
    min_val = min(y_true.min(), y_pred_baseline.min())
    max_val = max(y_true.max(), y_pred_baseline.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Baseline Predictions')
    ax1.set_title(f'Baseline - {label_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
   
    # Model scatter
    ax2.scatter(y_true, y_pred_model, alpha=0.6, s=30, color='blue')
    min_val = min(y_true.min(), y_pred_model.min())
    max_val = max(y_true.max(), y_pred_model.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Model Predictions')
    ax2.set_title(f'Model - {label_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    st.pyplot(fig)

def save_text_to_tempfile(name: str, content: str) -> str:
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

@st.cache_data(show_spinner=False, ttl=30)
def discover_checkpoints(search_root: str):
    try:
        return find_checkpoints_fixed(search_root)
    except Exception as e:
        st.error(f"Error discovering checkpoints: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_comprehensive_comparison_df(y_true_sel: np.ndarray, y_pred_baseline: np.ndarray,
                                   y_pred_model: np.ndarray, labels_sel: List[str]) -> pd.DataFrame:
    """Build comparison dataframe with both baseline and model predictions"""
    rows = []
    for j, name in enumerate(labels_sel):
        yt = y_true_sel[:, j]
        yp_baseline = y_pred_baseline[:, j]
        yp_model = y_pred_model[:, j]
       
        steps = np.arange(len(yt))
        err_baseline = yt - yp_baseline
        err_model = yt - yp_model
       
        for step, true_val, pred_base, pred_model, err_base, err_model_val in zip(
            steps, yt, yp_baseline, yp_model, err_baseline, err_model):
           
            rows.append({
                "label": name,
                "step": step,
                "true": true_val,
                "pred_baseline": pred_base,
                "pred_model": pred_model,
                "error_baseline": err_base,
                "error_model": err_model_val,
                "improvement": abs(err_base) - abs(err_model_val)
            })
   
    return pd.DataFrame(rows)

# ---------------------- Main App Interface ----------------------
st.sidebar.header(" Model and Comparison Settings")

# Configurable checkpoints root
st.sidebar.markdown("#####  Checkpoints Directory")
ckpt_root = st.sidebar.text_input(
    "Path to trained models",
    value=st.session_state.ckpt_root,
    help="Directory containing your pre-trained model checkpoints (from main.py training)",
    key="ckpt_root_input"
)

refresh_ckpts = st.sidebar.button(" Refresh Checkpoints", key="refresh_ckpts_btn")
if ckpt_root != st.session_state.ckpt_root or refresh_ckpts:
    st.session_state.ckpt_root = ckpt_root

# Discover available checkpoints
ckpts = discover_checkpoints(st.session_state.ckpt_root)
ckpt_labels = [lbl for lbl, _ in ckpts]
ckpt_map = {lbl: p for lbl, p in ckpts}

# Model selection
st.sidebar.markdown("#####  Model Selection")
if ckpt_labels:
    default_index = 0
    if st.session_state.selected_ckpt_label in ckpt_labels:
        default_index = ckpt_labels.index(st.session_state.selected_ckpt_label)
    selected_ckpt_label = st.sidebar.selectbox(
        "Choose a pre-trained model",
        options=ckpt_labels,
        index=default_index,
        help="Select a model to compare against baseline methods",
        key="ckpt_select"
    )
    st.session_state.selected_ckpt_label = selected_ckpt_label
    checkpoint_path = ckpt_map.get(selected_ckpt_label)
   
    st.sidebar.success(f" Selected: {selected_ckpt_label}")
else:
    st.sidebar.warning(" No trained models found!")
    st.sidebar.info("""
    **To get started:**
    1. Train models first: `python main.py --epochs 3 --use_gpu`
    2. Models will be saved in `./demo_outputs/`
    3. Click refresh above
    """)
    selected_ckpt_label = None
    checkpoint_path = None

# Baseline configuration
st.sidebar.markdown("#####  Baseline Configuration")
baseline_method = st.sidebar.selectbox(
    "Baseline prediction method",
    options=["last_value", "mean", "zero", "random"],
    index=0,
    help="Method for generating baseline predictions to compare against the model",
    key="baseline_method_select"
)

# Model configuration
st.sidebar.markdown("##### ⚡ Model Configuration")
col1, col2 = st.sidebar.columns(2)
with col1:
    seq_len = st.number_input(
        "Encoder length",
        min_value=8, max_value=512, value=96, step=8,
        key="seq_len_input",
        help="Historical time steps fed to encoder"
    )
    pred_len = st.number_input(
        "Forecast horizon",
        min_value=1, max_value=2048, value=96, step=1,
        key="pred_len_input",
        help="Future time steps to predict"
    )
with col2:
    label_len = st.number_input(
        "Decoder length",
        min_value=8, max_value=256, value=48, step=8,
        key="label_len_input",
        help="Past steps for decoder warm-up"
    )
    use_gpu = st.checkbox("Use GPU", value=False, key="use_gpu_checkbox")

# Action buttons
st.sidebar.markdown("#####  Actions")
run_comparison_btn = st.sidebar.button(" Run Comparison", type="primary", key="run_comparison_btn")
reset_btn = st.sidebar.button(" Reset", key="reset_btn")

if reset_btn:
    st.session_state.selected_labels = None
    st.session_state.comparison_results = None
    st.rerun()

# ---------------------- Data Input Section ----------------------
st.header(" Input Your Data")

tab1, tab2 = st.tabs([" Upload CSV File", "Type/Paste Data"])

uploaded_df: Optional[pd.DataFrame] = None

with tab1:
    st.subheader("Upload Your Dataset")
    st.markdown("Upload a CSV file with your time series data. Numeric columns will be used as targets.")
   
    up_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="First row should be headers. All numeric columns will be available for comparison.",
        key="csv_uploader"
    )
   
    if up_file:
        uploaded_path = os.path.join(tempfile.gettempdir(), up_file.name)
        with open(uploaded_path, "wb") as f:
            f.write(up_file.getbuffer())
        st.session_state.uploaded_path = uploaded_path
        st.session_state.dataset_name = up_file.name
        uploaded_df = load_csv_cached(uploaded_path)
       
        st.success(f" Successfully loaded **{up_file.name}**")
        st.write(f"**Dataset shape:** {uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} columns")
       
        # Show preview
        with st.expander(" Preview your data"):
            st.dataframe(uploaded_df.head(10))

with tab2:
    st.subheader("Enter Data Manually")
    st.markdown("Paste your data in CSV format (comma-separated values).")
   
    default_name = st.session_state.dataset_name or "my_dataset.csv"
    typed_name = st.text_input("Dataset name", value=default_name, key="dataset_name_input")
   
    default_data = """timestamp,feature1,feature2,target1,target2
2024-01-01,1.2,3.4,5.6,7.8
2024-01-02,1.3,3.5,5.7,7.9
2024-01-03,1.4,3.6,5.8,8.0
2024-01-04,1.5,3.7,5.9,8.1
2024-01-05,1.6,3.8,6.0,8.2"""
   
    typed_text = st.text_area("CSV Data", height=200, value=default_data, key="data_text_area")
   
    add_typed = st.button("Use this data", key="add_typed_btn")
    if add_typed and typed_text.strip():
        uploaded_path = save_text_to_tempfile(typed_name, typed_text)
        st.session_state.uploaded_path = uploaded_path
        st.session_state.dataset_name = typed_name
        uploaded_df = load_csv_cached(uploaded_path)
        st.success(f" Loaded dataset with {uploaded_df.shape[0]} rows")

# Restore previous data if available
if uploaded_df is None and st.session_state.uploaded_path and os.path.isfile(st.session_state.uploaded_path):
    try:
        uploaded_df = load_csv_cached(st.session_state.uploaded_path)
        st.info(f" Using previously loaded dataset: {st.session_state.dataset_name}")
    except Exception:
        uploaded_df = None

# ---------------------- Label Selection ----------------------
if uploaded_df is not None:
    numeric_cols = get_numeric_columns(uploaded_df)
   
    if not numeric_cols:
        st.error(" No numeric columns found in your data!")
        st.info("Please upload a CSV file with at least one numeric column to use as prediction targets.")
        st.stop()

    st.header(" Select Prediction Targets")
    st.markdown("Choose which numeric columns to evaluate. The model will make predictions for each selected column.")
   
    # Label selection with persistence
    raw_default = st.session_state.selected_labels if st.session_state.selected_labels is not None else numeric_cols
   
    if isinstance(raw_default, str):
        raw_default_list = [raw_default]
    else:
        try:
            raw_default_list = list(raw_default)
        except Exception:
            raw_default_list = []
   
    # Keep only valid defaults
    valid_defaults = [col for col in raw_default_list if col in numeric_cols]
    if not valid_defaults and numeric_cols:
        valid_defaults = numeric_cols[:2]  # Default to first 2 columns

    select_cols = st.multiselect(
        "Target columns for comparison",
        options=numeric_cols,
        default=valid_defaults,
        help="Select the numeric columns you want to evaluate. The model will predict future values for each selected column.",
        key="label_multiselect"
    )
    st.session_state.selected_labels = select_cols

    if len(select_cols) == 0:
        st.warning(" Please select at least one target column to compare.")
        st.stop()

    selected_indices = [numeric_cols.index(c) for c in select_cols]

    # Show selected targets info
    st.success(f" Selected {len(select_cols)} target(s): {', '.join(select_cols)}")

    # ---------------------- Run Comparison ----------------------
    if run_comparison_btn:
        if not checkpoint_path:
            st.error(" No pre-trained model selected!")
            st.info("Please select a model from the sidebar or train models first using `python main.py`")
        else:
            try:
                with st.spinner(" Running model inference and generating baseline comparisons..."):
                    # Run model inference
                    y_true, y_pred_model = run_inference_csv_fixed(
                        checkpoint_path=checkpoint_path,
                        dataset_key="custom",
                        csv_path=st.session_state.uploaded_path,
                        seq_len=int(seq_len),
                        label_len=int(label_len),
                        pred_len=int(pred_len),
                        use_gpu=use_gpu,
                    )

                    # Generate baseline predictions
                    y_pred_baseline = generate_baseline_predictions(y_true, method=baseline_method)

                    # Slice to selected columns
                    y_true_sel = y_true[:, selected_indices]
                    y_pred_model_sel = y_pred_model[:, selected_indices]
                    y_pred_baseline_sel = y_pred_baseline[:, selected_indices]
                    labels_sel = select_cols

                    # Store comprehensive results
                    st.session_state.comparison_results = {
                        'y_true': y_true_sel,
                        'y_pred_baseline': y_pred_baseline_sel,
                        'y_pred_model': y_pred_model_sel,
                        'labels': labels_sel,
                        'baseline_method': baseline_method,
                        'model_name': selected_ckpt_label
                    }

                st.success(" Comparison complete! Showing results below.")

            except Exception as e:
                st.error(f" Error during comparison: {e}")
                with st.expander(" Show technical details"):
                    st.code(traceback.format_exc())

# ---------------------- Results Display ----------------------
if st.session_state.comparison_results:
    results = st.session_state.comparison_results
    y_true_sel = results['y_true']
    y_pred_baseline_sel = results['y_pred_baseline']
    y_pred_model_sel = results['y_pred_model']
    labels_sel = results['labels']
   
    st.header(" Comparison Results")
   
    # Summary header
    st.subheader("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric("Model", results['model_name'])
    with col2:
        st.metric("Baseline Method", results['baseline_method'].replace('_', ' ').title())
    with col3:
        st.metric("Targets", len(labels_sel))
    with col4:
        st.metric("Horizon", f"{pred_len} steps")

    # Per-label detailed comparison
    st.subheader(" Detailed Metrics by Target")
   
    all_metrics_baseline = []
    all_metrics_model = []
   
    for j, label_name in enumerate(labels_sel):
        st.markdown(f"#### {label_name}")
       
        yt_col = y_true_sel[:, j]
        yp_baseline_col = y_pred_baseline_sel[:, j]
        yp_model_col = y_pred_model_sel[:, j]
       
        # Compute metrics
        metrics_baseline = compute_comprehensive_metrics(yt_col, yp_baseline_col, label_name, "baseline")
        metrics_model = compute_comprehensive_metrics(yt_col, yp_model_col, label_name, "model")
       
        all_metrics_baseline.append(metrics_baseline)
        all_metrics_model.append(metrics_model)
       
        # Display comparison
        render_comparison_metrics(metrics_baseline, metrics_model)
       
        # Visualizations
        st.markdown("##### Visual Comparisons")
        col_viz1, col_viz2 = st.columns(2)
       
        with col_viz1:
            plot_comparison_timeseries(yt_col, yp_baseline_col, yp_model_col, label_name)
       
        with col_viz2:
            plot_comparison_scatter(yt_col, yp_baseline_col, yp_model_col, label_name)
       
        st.markdown("---")

    # Aggregate comparison
    st.subheader("Aggregate Performance Summary")
   
    # Convert to dataframes for easier analysis
    df_baseline = pd.DataFrame(all_metrics_baseline)
    df_model = pd.DataFrame(all_metrics_model)
   
    # Display aggregate metrics
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("##### Baseline Statistics")
        st.dataframe(df_baseline[['mae', 'rmse', 'r2']].describe().round(4))
   
    with col2:
        st.markdown("#####  Model Statistics")
        st.dataframe(df_model[['mae', 'rmse', 'r2']].describe().round(4))
   
    # Improvement analysis - FIXED SECTION
    st.subheader(" Performance Improvement Analysis")
   
    improvement_data = []
    for base, model in zip(all_metrics_baseline, all_metrics_model):
        imp = {
            'Target': base['label'],
            'MAE Improvement %': (base['mae'] - model['mae']) / base['mae'] * 100 if base['mae'] != 0 else 0,
            'RMSE Improvement %': (base['rmse'] - model['rmse']) / base['rmse'] * 100 if base['rmse'] != 0 else 0,
            'R² Improvement': (model['r2'] - base['r2']) * 100,
        }
        improvement_data.append(imp)
   
    df_improvement = pd.DataFrame(improvement_data)
   
    # FIXED: Completely simplified styling without lambda functions
    try:
        # Format the numbers first
        formatted_df = df_improvement.copy()
        formatted_df['MAE Improvement %'] = formatted_df['MAE Improvement %'].apply(lambda x: f"{x:.2f}%")
        formatted_df['RMSE Improvement %'] = formatted_df['RMSE Improvement %'].apply(lambda x: f"{x:.2f}%")
        formatted_df['R² Improvement'] = formatted_df['R² Improvement'].apply(lambda x: f"{x:.2f}")
       
        # Display without pandas styling to avoid the error
        st.dataframe(formatted_df)
       
        # Show color-coded summary instead
        st.markdown("#####  Color-coded Summary:")
        for _, row in df_improvement.iterrows():
            mae_imp = row['MAE Improvement %']
            rmse_imp = row['RMSE Improvement %']
            r2_imp = row['R² Improvement']
           
            mae_color = "🟢" if mae_imp > 0 else "🔴" if mae_imp < 0 else "⚪"
            rmse_color = "🟢" if rmse_imp > 0 else "🔴" if rmse_imp < 0 else "⚪"
            r2_color = "🟢" if r2_imp > 0 else "🔴" if r2_imp < 0 else "⚪"
           
            st.write(f"**{row['Target']}:** {mae_color} MAE: {mae_imp:+.2f}% | {rmse_color} RMSE: {rmse_imp:+.2f}% | {r2_color} R²: {r2_imp:+.2f}")
       
    except Exception as style_error:
        # Ultimate fallback: Show simple table
        st.warning("⚠️ Using simplified display for improvement table:")
        st.dataframe(df_improvement.round(4))
   
    # Download comprehensive results
    st.subheader(" Download Results")
   
    comparison_df = build_comprehensive_comparison_df(
        y_true_sel, y_pred_baseline_sel, y_pred_model_sel, labels_sel
    )
   
    col_dl1, col_dl2, col_dl3 = st.columns(3)
   
    with col_dl1:
        st.download_button(
            label=" Download Comparison Data",
            data=comparison_df.to_csv(index=False),
            file_name="model_vs_baseline_comparison.csv",
            mime="text/csv",
            key="download_comparison_btn"
        )
   
    with col_dl2:
        # Combine metrics for download
        metrics_combined = []
        for base, model in zip(all_metrics_baseline, all_metrics_model):
            combined = {**base, **{f"model_{k}": v for k, v in model.items() if k != 'label'}}
            metrics_combined.append(combined)
       
        df_metrics_combined = pd.DataFrame(metrics_combined)
        st.download_button(
            label=" Download Metrics Summary",
            data=df_metrics_combined.to_csv(index=False),
            file_name="comparison_metrics.csv",
            mime="text/csv",
            key="download_metrics_btn"
        )
   
    with col_dl3:
        # Create a summary report
        summary_report = f"""
        Model vs Baseline Comparison Report
        ===================================
       
        Model: {results['model_name']}
        Baseline Method: {results['baseline_method']}
        Targets Evaluated: {len(labels_sel)}
        Forecast Horizon: {pred_len} steps
       
        Summary of Improvements:
        {df_improvement.to_string(index=False)}
       
        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
       
        st.download_button(
            label=" Download Summary Report",
            data=summary_report,
            file_name="comparison_summary.txt",
            mime="text/plain",
            key="download_summary_btn"
        )

else:
    # Show welcome message when no comparison has been run
    st.info("""
    ##  Welcome to the Model Comparison App!
   
    **To get started:**
   
    1. ** Upload your data** using one of the tabs above
    2. ** Select a trained model** from the sidebar  
    3. ** Choose target columns** for comparison
    4. ** Configure model settings** (sequence lengths, etc.)
    5. ** Click 'Run Comparison'** to see results!
   
    **Don't see any models?** Make sure to train them first:
    ```bash
    python main.py --epochs 3 --use_gpu
    ```
    """)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Model Comparison App • Compare pre-trained models against baseline methods on your unseen data
    </div>
    """,
    unsafe_allow_html=True
)