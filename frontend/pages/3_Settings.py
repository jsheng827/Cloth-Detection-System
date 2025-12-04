import os
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Make sure we can import from the existing `py/` modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PY_DIR = os.path.join(BASE_DIR, "py")
if PY_DIR not in sys.path:
    sys.path.append(PY_DIR)

from model_manager import (  # type: ignore
    add_model,
    delete_model,
    get_all_models,
    get_models_by_type,
    MODEL_DIR,
    model_exists,
)

st.set_page_config(
    page_title="Settings",
    layout="wide",
    page_icon="⚙️",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .dashboard-root {
        background-color: #0f0f0f;
        padding: 1.25rem 2rem;
        color: #f5f5f5;
    }
    .model-card {
        background: #181818;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }
    .model-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #f2f2f2;
    }
    body {
        background-color: #0f0f0f;
        color: #f5f5f5;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Settings")
st.caption("Manage your models and system configuration")

# Model Section
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown('<div class="model-header">📦 Model Management</div>', unsafe_allow_html=True)

# Tabs for different model types
tab1, tab2, tab3, tab4 = st.tabs(
    ["Detection Models", "Re-ID Models", "Cloth Models", "Shoe Models"]
)

# Helper function to render upload form
def render_upload_form(model_type: str, model_type_label: str):
    """Render upload form for a specific model type."""
    st.subheader(f"Upload New {model_type_label}")
    
    with st.form(f"upload_{model_type}_form", clear_on_submit=True):
        model_name = st.text_input(
            "Model Name",
            help=f"Enter a friendly name for this {model_type_label.lower()}",
            key=f"name_{model_type}",
        )
        
        uploaded_file = st.file_uploader(
            f"Choose {model_type_label} File",
            type=["pt", "pth", "onnx", "engine", "plan"],
            help=f"Upload a {model_type_label.lower()} file (.pt, .pth, .onnx, .engine, .plan)",
            key=f"file_{model_type}",
        )
        
        submit = st.form_submit_button("Upload Model", type="primary")
        
        if submit:
            if not model_name or not model_name.strip():
                st.error("Please enter a model name")
                return
            
            if not uploaded_file:
                st.error("Please select a file to upload")
                return
            
            # Check if model name already exists for this type
            if model_exists(model_name.strip(), model_type):
                st.error(f"A {model_type_label.lower()} with name '{model_name.strip()}' already exists. Please choose a different name.")
                return
            
            try:
                # Save file to model directory
                file_extension = Path(uploaded_file.name).suffix
                safe_filename = f"{model_name.strip().replace(' ', '_')}{file_extension}"
                file_path = MODEL_DIR / safe_filename
                
                # Write file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to metadata
                relative_path = f"./model/{safe_filename}"
                add_model(
                    model_name=model_name.strip(),
                    model_type=model_type,
                    file_path=relative_path,
                    original_filename=uploaded_file.name,
                )
                
                st.success(f"✅ Model '{model_name.strip()}' uploaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to upload model: {e}")

# Helper function to render model list
def render_model_list(model_type: str, model_type_label: str):
    """Render list of uploaded models for a specific type."""
    models = get_models_by_type(model_type)
    
    if not models:
        st.info(f"No {model_type_label.lower()} uploaded yet. Use the form above to upload one.")
        return
    
    st.subheader(f"Uploaded {model_type_label}")
    
    for model in models:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                st.write(f"**{model['name']}**")
                st.caption(f"File: {model['original_filename']}")
            
            with col2:
                st.caption(f"Path: `{model['path']}`")
                if "uploaded_at" in model:
                    st.caption(f"Uploaded: {model['uploaded_at'][:10]}")
            
            with col3:
                # Check if file exists
                file_path = MODEL_DIR / Path(model['path']).name
                if file_path.exists():
                    st.success("✓ File exists")
                else:
                    st.warning("⚠ File missing")
            
            with col4:
                if st.button("Delete", key=f"delete_{model_type}_{model['name']}", type="secondary"):
                    if delete_model(model['name'], model_type):
                        st.success(f"Deleted {model['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete model")
            
            st.divider()

# Detection Models Tab
with tab1:
    render_upload_form("detection", "Detection Model")
    st.markdown("---")
    render_model_list("detection", "Detection Models")

# Re-ID Models Tab
with tab2:
    render_upload_form("reid", "Re-ID Model")
    st.markdown("---")
    render_model_list("reid", "Re-ID Models")

# Cloth Models Tab
with tab3:
    render_upload_form("cloth", "Cloth Model")
    st.markdown("---")
    render_model_list("cloth", "Cloth Models")

# Shoe Models Tab
with tab4:
    render_upload_form("shoe", "Shoe Model")
    st.markdown("---")
    render_model_list("shoe", "Shoe Models")

st.markdown("</div>", unsafe_allow_html=True)

# Instructions
st.markdown("---")
with st.expander("ℹ️ How to use Model Management"):
    st.markdown("""
    **Uploading Models:**
    1. Navigate to the appropriate tab (Detection, Re-ID, Cloth, or Shoe)
    2. Enter a friendly name for your model
    3. Select the model file (.pt, .pth, .onnx, .engine, or .plan)
    4. Click "Upload Model"
    
    **Using Models:**
    - Uploaded models will appear in dropdown menus on the Real-Time Dashboard
    - Select models from the sidebar configuration instead of typing paths
    
    **Deleting Models:**
    - Click the "Delete" button next to any model to remove it
    - This will delete both the file and its metadata
    """)


