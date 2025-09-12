# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import torch

# --- CORE LOGIC IMPORTS ---
from src import predictor
from src.baseline import BaselineIDS

# --- CONFIGURATION ---
MODEL_DIR = "models"
DATA_DIR = "data/processed"
ROBUST_MODEL_PATH = os.path.join(MODEL_DIR, "robust_model.pt")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")
COLUMNS_PATH = os.path.join(DATA_DIR, "feature_columns.json")

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_assets():
    """Loads all necessary assets into memory. Cached for performance."""
    if not all([os.path.exists(p) for p in [ROBUST_MODEL_PATH, SCALER_PATH, COLUMNS_PATH]]):
        return None, None, None
    
    with open(COLUMNS_PATH, 'r') as f:
        feature_columns = json.load(f)
    model = BaselineIDS(input_features=len(feature_columns))
    model.load_state_dict(torch.load(ROBUST_MODEL_PATH, map_location='cpu'))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    
    return model, scaler, feature_columns

def preprocess_dataframe(df, scaler, required_columns):
    """
    Takes a raw dataframe and preprocesses it for the model.
    This now includes the critical label encoding step.
    """
    # Handle infinities and NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in required_columns:
        if df[col].isnull().any():
            col_index = required_columns.index(col)
            if hasattr(scaler, 'mean_'):
                 fill_value = scaler.mean_[col_index]
                 df[col].fillna(fill_value, inplace=True)

    features = df[required_columns]
    scaled_features = scaler.transform(features)
    processed_df = pd.DataFrame(scaled_features, columns=required_columns)
    
    # --- DEFINITIVE FIX ---
    # The 'Label' column must be numerically encoded to match the model's output.
    # This logic is now synchronized with our main preprocessing script.
    if 'Label' in df.columns:
        # Ensure we handle potential whitespace and case issues robustly.
        labels_series = df['Label'].astype(str).str.strip().str.lower()
        processed_df['Label'] = labels_series.apply(lambda x: 1 if x == 'malicious' else 0).values
    # --- END FIX ---
        
    return processed_df

def fuzzy_match_columns(user_columns, required_columns):
    """Performs an intelligent, heuristic mapping."""
    user_map = {col.lower().replace(" ", "").replace("_", ""): col for col in user_columns}
    mapping = {}
    unmapped = []

    for req_col in required_columns:
        req_key = req_col.lower().replace(" ", "").replace("_", "")
        if req_key in user_map:
            mapping[req_col] = user_map[req_key]
        else:
            unmapped.append(req_col)
            
    return mapping, unmapped

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="5G IDS Security Panel")

st.title("🛡️ 5G IDS Security Control Panel")
st.markdown("""
Welcome to the 5G Intrusion Detection System Dashboard. 
This tool leverages a robust, adversarially-trained deep learning model to analyze network traffic and assess its security posture.
""")

model, scaler, feature_columns = load_assets()

if not all([model, scaler, feature_columns]):
    st.error("FATAL ERROR: Critical assets not found. Please run the training pipeline.")
    st.stop()

uploaded_file = st.file_uploader("Upload Your Network Traffic CSV for Analysis", type="csv")

if uploaded_file is not None:
    try:
        st.session_state.user_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

if 'user_df' in st.session_state:
    user_df = st.session_state.user_df
    user_columns = user_df.columns.tolist()

    auto_mapping, unmapped_reqs = fuzzy_match_columns(user_columns, feature_columns)
    final_mapping = auto_mapping.copy()

    if unmapped_reqs:
        st.warning(f"⚠️ **Action Required:** We couldn't automatically map **{len(unmapped_reqs)}** column(s). Please resolve the ambiguities below.")
        claimed_user_cols = set(auto_mapping.values())
        available_options = [col for col in user_columns if col not in claimed_user_cols]

        for req_col in unmapped_reqs:
            selected_col = st.selectbox(f"Map **'{req_col}'** to:", options=["-"] + available_options, key=f"map_{req_col}")
            if selected_col != "-":
                final_mapping[req_col] = selected_col
    
    if st.button("Confirm Mappings & Run Analysis"):
        if len(final_mapping) != len(feature_columns):
            st.error("Mapping Error: Not all required features have been mapped.")
        elif len(set(final_mapping.values())) != len(final_mapping.values()):
            st.error("Mapping Error: A column from your file has been mapped to multiple required features. Please ensure mappings are unique.")
        else:
            with st.spinner("Analyzing traffic..."):
                try:
                    # Work on a copy of the user's dataframe.
                    analysis_df = user_df.copy()
                    # Rename the columns based on the final mapping.
                    analysis_df.rename(columns={v: k for k, v in final_mapping.items()}, inplace=True)
                    
                    # Preprocess the dataframe, which now includes label encoding.
                    processed_df = preprocess_dataframe(analysis_df, scaler, feature_columns)

                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, "analysis_temp.csv")
                    processed_df.to_csv(temp_path, index=False)
                    
                    metrics = predictor.run_evaluation(
                        model_path=ROBUST_MODEL_PATH,
                        data_path=temp_path,
                        columns_path=COLUMNS_PATH
                    )
                    
                    if metrics is None:
                        raise ValueError("The evaluation script returned no metrics. This is often due to a missing 'Label' column in the processed data.")

                    st.session_state.results = {"metrics": metrics, "filename": uploaded_file.name, "rows": len(user_df)}
                    
                except Exception as e:
                    st.session_state.results = {"error": str(e)}

if 'results' in st.session_state:
    results = st.session_state.results
    st.header("Analysis Results")

    if "error" in results:
        st.error(f"An unexpected error occurred during analysis: {results['error']}")
    else:
        metrics = results['metrics']
        f1 = metrics['f1']

        if f1 > 0.6:
            verdict, color = "SECURE", "green"
            message = f"The model shows strong performance (F1-Score: {f1:.4f}). Traffic patterns appear normal and the network exhibits resilience."
        elif f1 > 0.4:
            verdict, color = "CAUTION", "orange"
            message = f"The model shows moderate performance (F1-Score: {f1:.4f}). Some anomalous patterns were detected."
        else:
            verdict, color = "COMPROMISED", "red"
            message = f"The model shows poor performance (F1-Score: {f1:.4f}). The network is likely under duress."

        st.markdown(f"### Verdict: <span style='color:{color};'>{verdict}</span>", unsafe_allow_html=True)
        st.write(message)
        
        st.subheader("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("F1-Score", f"{metrics['f1']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")

        st.subheader("Analysis Summary")
        summary_data = {"Filename": [results['filename']], "Total Flows Analyzed": [results['rows']]}
        st.table(pd.DataFrame(summary_data))

