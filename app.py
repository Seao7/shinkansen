import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from PIL import Image
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import zipfile
from pathlib import Path


# ============================================
# HSC Data Conversion Functions
# ============================================


def read_hsc_data(base_path):
    """
    Load HSC data (.txt + .dat) and convert to radiance

    Parameters:
    -----------
    base_path : str
        Base file path (without extension)

    Returns:
    --------
    rad : np.ndarray
        Radiance data (B, H, W)
    wavelengths : np.ndarray
        Wavelength data
    """
    # 1. Get parameters from .txt file
    W, H, B, ABand, Ag, t, wavelengths = read_whb(base_path + ".txt")

    # 2. Convert .dat to NumPy array
    dn = read_dat_bil(base_path + ".dat", W, H, B)

    # 3. Convert to radiance
    rad = to_radiance_fixed(dn, ABand, Ag, t)

    return rad, wavelengths


def read_whb(txt_path):
    """Get W,H,B and parameters from text file"""
    with open(txt_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Line 2 contains W,H,B
    W, H, B = map(int, map(float, lines[1].split(",")[:3]))

    # Extract non-comment lines only
    non_comment = [l for l in lines if not l.startswith("#")]
    band_rows = non_comment[1:1+B]  # B rows of band data

    wavelengths = [float(r.split(",")[0]) for r in band_rows]
    ABand = [float(r.split(",")[1]) for r in band_rows]
    Ag    = [float(r.split(",")[2]) for r in band_rows]
    t     = [float(r.split(",")[3]) for r in band_rows]

    return W, H, B, np.array(ABand), np.array(Ag), np.array(t), np.array(wavelengths)


def read_dat_bil(dat_path, W, H, B):
    """Convert BIL format .dat file to (B,H,W) shape"""
    buf = np.fromfile(dat_path, dtype=np.uint8)
    out = np.empty((B, H, W), dtype=np.uint8)
    k = 0
    for h in range(H):
        for b in range(B):
            out[b, h, :] = buf[k:k+W]
            k += W
    return out


def to_radiance_fixed(dn, ABand, Ag, t, AFn=1.0):
    """
    Convert digital numbers to radiance
    R = Scamave * a * t^b * (Do / Ag) * ABand * AFn
    """
    Scamave, a, b = 0.99959, 2.0035, -0.984
    B = dn.shape[0]

    ABand = np.asarray(ABand, dtype=np.float32)
    Ag    = np.asarray(Ag,    dtype=np.float32)
    t     = np.asarray(t,     dtype=np.float32)

    if len(ABand) < B or len(Ag) < B or len(t) < B:
        raise ValueError(f"Insufficient parameters: bands={B}, ABand={len(ABand)}, Ag={len(Ag)}, t={len(t)}")

    # Reshape to match band count
    ABand = ABand[:B].reshape(B, 1, 1)
    Ag    = Ag[:B].reshape(B, 1, 1)
    t     = t[:B].reshape(B, 1, 1)

    return (dn.astype(np.float32) * (Scamave * a) * (t ** b) / Ag) * ABand * AFn


@st.cache_resource
def load_model(component_type):
    """Load the trained model for specified component type"""
    model_path = f'oil_detection_model_{component_type.lower()}.pkl'
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file '{model_path}' not found in the current directory.")
        st.info(f"The app will run in placeholder mode for {component_type}. To use a trained model, place '{model_path}' in the same directory as this script.")
        return None
    
    try:
        artifacts = joblib.load(model_path)
        st.success(f"✅ {component_type} model loaded: {artifacts['model_name']}")
        return artifacts
    except Exception as e:
        st.error(f"❌ Error loading {component_type} model: {str(e)}")
        return None


def predict_oil_pixels(rad_data, wavelengths, model_artifacts):
    """
    Use the trained model to predict oil pixels
    
    Parameters:
    -----------
    rad_data : np.ndarray
        Radiance data (B, H, W)
    wavelengths : np.ndarray
        Wavelength data
    model_artifacts : dict
        Dictionary containing model, scaler, and other artifacts
    
    Returns:
    --------
    oil_mask : np.ndarray
        Binary mask where 1 indicates oil (H, W)
    probability_map : np.ndarray
        Probability map for oil class (H, W) if available
    """
    model = model_artifacts['model']
    background = model_artifacts.get('background', None)
    scaler = model_artifacts['scaler']
    oil_label_index = model_artifacts['oil_label_index']
    
    # Get shape
    B, H, W = rad_data.shape
    
    # Apply background correction if available
    if background is not None and 'radiance' in background:
        bg_radiance = background['radiance'].reshape(B, 1, 1)
        bg_radiance = np.where(bg_radiance == 0, 1.0, bg_radiance)  # Replace zeros with 1
        rad_data = rad_data / bg_radiance
        rad_data = np.nan_to_num(rad_data, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Reshape data: (B, H, W) -> (H*W, B)
    pixels = rad_data.reshape(B, -1).T  # Shape: (H*W, B)
    
    # Scale the data
    pixels_scaled = scaler.transform(pixels)
    
    # Predict
    predictions = model.predict(pixels_scaled)
    
    # Create binary mask (1 for oil, 0 for not oil)
    oil_mask = (predictions == oil_label_index).astype(np.float32)
    
    # Reshape back to image
    oil_mask = oil_mask.reshape(H, W)
    
    # Get probability map if model supports it
    probability_map = None
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(pixels_scaled)
            # Get probability for oil class
            oil_probabilities = probabilities[:, oil_label_index]
            probability_map = oil_probabilities.reshape(H, W)
        elif hasattr(model, 'decision_function'):
            # For SVM and similar models
            decision_values = model.decision_function(pixels_scaled)
            if len(decision_values.shape) > 1:
                decision_values = decision_values[:, oil_label_index]
            # Normalize to 0-1 range
            decision_values = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
            probability_map = decision_values.reshape(H, W)
    except Exception as e:
        st.warning(f"Could not generate probability map: {str(e)}")
    
    return oil_mask, probability_map


def find_matching_pairs(folder_path):
    """Find all matching .dat and .txt file pairs in a folder"""
    dat_files = list(Path(folder_path).glob("*.dat"))
    pairs = []
    
    for dat_file in dat_files:
        txt_file = dat_file.with_suffix(".txt")
        if txt_file.exists():
            pairs.append((str(dat_file), str(txt_file)))
    
    return pairs


def generate_combined_pdf_report(results_list, filename, model_name=None, component_type=None):
    """Generate a combined PDF report with all results"""
    
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        title = f'Hyperspectral Oil Detection Report - {component_type}' if component_type else 'Hyperspectral Oil Detection Report'
        fig.suptitle(title, fontsize=24, fontweight='bold', y=0.95)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.5, 0.85, f'Generated: {timestamp}', ha='center', fontsize=14)
        
        total_files = len(results_list)
        avg_oil_percentage = np.mean([r['oil_percentage'] for r in results_list])
        
        summary_text = f"""
        Batch Analysis Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Total Files Processed: {total_files}
        Average Oil Coverage: {avg_oil_percentage:.2f}%
        Component Type: {component_type}
        Model Used: {model_name if model_name else 'Placeholder'}
        
        Files Analyzed:
        """
        
        for i, result in enumerate(results_list[:10], 1):  # Show first 10
            summary_text += f"\n        {i}. {result['file_name']} - {result['oil_percentage']:.2f}%"
        
        if total_files > 10:
            summary_text += f"\n        ... and {total_files - 10} more files"
        
        plt.text(0.1, 0.7, summary_text, fontsize=11, verticalalignment='top', 
                family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        for result in results_list:
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(f"File: {result['file_name']}", fontsize=18, fontweight='bold', y=0.95)
            
            model_info = f"\nModel Used: {model_name}" if model_name else "\nModel Used: Placeholder"
            file_summary = f"""
            Analysis Details
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            {model_info}
            Component Type: {component_type}
            
            RGB Composite Settings:
            • Red Channel: {result['wavelengths'][result['red_idx']]:.1f} nm (Band {result['red_idx']})
            • Green Channel: {result['wavelengths'][result['green_idx']]:.1f} nm (Band {result['green_idx']})
            • Blue Channel: {result['wavelengths'][result['blue_idx']]:.1f} nm (Band {result['blue_idx']})
            
            Detection Results:
            • Oil Coverage: {result['oil_percentage']:.2f}%
            • Total Pixels: {result['oil_mask'].size:,}
            • Oil Pixels Detected: {int(result['oil_mask'].sum()):,}
            """
            
            plt.text(0.1, 0.75, file_summary, fontsize=11, verticalalignment='top', 
                    family='monospace')
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.imshow(result['rgb_img'])
            ax.set_title(f"{result['file_name']} - RGB Composite", fontsize=14, fontweight='bold', pad=20)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.imshow(result['rgb_img'])
            ax.imshow(result['oil_mask'], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f"{result['file_name']} - Oil Detection", fontsize=14, fontweight='bold', pad=20)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if result['probability_map'] is not None:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                im = ax.imshow(result['probability_map'], cmap='hot', vmin=0, vmax=1)
                ax.set_title(f"{result['file_name']} - Probability Map", fontsize=14, fontweight='bold', pad=20)
                ax.axis('off')
                plt.colorbar(im, ax=ax, label='Probability', fraction=0.046, pad=0.04)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    return pdf_path


def generate_pdf_report(rgb_img, oil_mask, wavelengths, red_idx, green_idx, blue_idx, 
                       oil_percentage, filename, model_name=None, probability_map=None, 
                       component_type=None):
    """Generate a PDF report with analysis results"""
    
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        title = f'Hyperspectral Oil Detection Report - {component_type}' if component_type else 'Hyperspectral Oil Detection Report'
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.5, 0.85, f'Generated: {timestamp}', ha='center', fontsize=12)
        
        # Summary statistics
        model_info = f"\nModel Used: {model_name}" if model_name else "\nModel Used: Placeholder (Random)"
        component_info = f"\nComponent Type: {component_type}" if component_type else ""
        summary_text = f"""
        Analysis Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {model_info}{component_info}
        
        RGB Composite Settings:
        • Red Channel: {wavelengths[red_idx]:.1f} nm (Band {red_idx})
        • Green Channel: {wavelengths[green_idx]:.1f} nm (Band {green_idx})
        • Blue Channel: {wavelengths[blue_idx]:.1f} nm (Band {blue_idx})
        
        Detection Results:
        • Oil Coverage: {oil_percentage:.2f}%
        • Total Pixels: {oil_mask.size:,}
        • Oil Pixels Detected: {int(oil_mask.sum()):,}
        
        Available Wavelengths:
        {', '.join([f'{w:.1f} nm' for w in wavelengths])}
        """
        
        plt.text(0.1, 0.6, summary_text, fontsize=11, verticalalignment='top', 
                family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: RGB Composite
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.imshow(rgb_img)
        ax.set_title('RGB Composite Image', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Oil Detection Overlay
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.imshow(rgb_img)
        ax.imshow(oil_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax.set_title('Oil Detection Overlay', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Probability map if available
        if probability_map is not None:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            im = ax.imshow(probability_map, cmap='hot', vmin=0, vmax=1)
            ax.set_title('Oil Probability Map', fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Probability', fraction=0.046, pad=0.04)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page 5: Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original RGB Composite", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(rgb_img)
        axes[1].imshow(oil_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title("With Oil Detection Overlay", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle('Comparison View', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    return pdf_path


def process_single_file(base_path, component_type, model_artifacts, file_index=0, total_files=1):
    """Process a single file pair and return results"""
    try:
        # Read data
        rad_data, wavelengths = read_hsc_data(base_path)
        
        # Determine default wavelengths
        def find_closest_wavelength(target, available_wls):
            idx = np.argmin(np.abs(available_wls - target))
            return float(available_wls[idx])
        
        if wavelengths.max() >= 650:
            default_red_wl = find_closest_wavelength(650, wavelengths)
        else:
            default_red_wl = float(wavelengths.max())
        
        if wavelengths.min() <= 550 <= wavelengths.max():
            default_green_wl = find_closest_wavelength(550, wavelengths)
        else:
            mid_idx = len(wavelengths) // 2
            default_green_wl = float(wavelengths[mid_idx])
        
        if wavelengths.min() <= 450:
            default_blue_wl = find_closest_wavelength(450, wavelengths)
        else:
            default_blue_wl = float(wavelengths.min())
        
        # Find wavelength indices
        red_idx = int(np.where(wavelengths == default_red_wl)[0][0])
        green_idx = int(np.where(wavelengths == default_green_wl)[0][0])
        blue_idx = int(np.where(wavelengths == default_blue_wl)[0][0])
        
        # Create RGB composite
        red_ch = rad_data[red_idx]
        green_ch = rad_data[green_idx]
        blue_ch = rad_data[blue_idx]
        
        def normalize(channel):
            min_val, max_val = np.min(channel), np.max(channel)
            return (channel - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(channel)
        
        rgb_img = np.stack([
            normalize(red_ch), 
            normalize(green_ch), 
            normalize(blue_ch)
        ], axis=2)
        
        # Run detection
        if model_artifacts is not None:
            oil_mask, probability_map = predict_oil_pixels(rad_data, wavelengths, model_artifacts)
        else:
            oil_mask = np.random.rand(*rgb_img.shape[:2])
            oil_mask = (oil_mask > 0.85).astype(np.float32)
            probability_map = None
        
        oil_percentage = (oil_mask.sum() / oil_mask.size) * 100
        
        return {
            'success': True,
            'rad_data': rad_data,
            'wavelengths': wavelengths,
            'rgb_img': rgb_img,
            'oil_mask': oil_mask,
            'probability_map': probability_map,
            'oil_percentage': oil_percentage,
            'red_idx': red_idx,
            'green_idx': green_idx,
            'blue_idx': blue_idx
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ============================================
# Streamlit App
# ============================================

# ---- App Title ----
st.set_page_config(page_title="HSC Oil Detection", layout="wide")
st.title("🛢️ Hyperspectral Oil Detection Tool")

st.write("""
This Streamlit app lets you upload hyperspectral files (.dat + .txt) for Axle or Motor components,
visualize RGB composites, and run oil detection using trained machine learning models.
""")

# Initialize session state
if 'oil_results' not in st.session_state:
    st.session_state.oil_results = []
if 'current_component' not in st.session_state:
    st.session_state.current_component = "Axle"

# ---- Component Type Selection ----
st.sidebar.header("⚙️ Configuration")
component_type = st.sidebar.radio(
    "Select Component Type",
    options=["Axle", "Motor"],
    index=0 if st.session_state.current_component == "Axle" else 1,
    help="Choose the type of component you're analyzing"
)

# Update component type in session state
if component_type != st.session_state.current_component:
    st.session_state.current_component = component_type
    load_model.clear()

# ---- Load Model for Selected Component ----
with st.spinner(f"Loading {component_type} model..."):
    model_artifacts = load_model(component_type)

if model_artifacts is not None:
    with st.sidebar:
        st.header(f"📊 {component_type} Model Info")
        st.write(f"**Model:** {model_artifacts['model_name']}")
        st.write(f"**Classes:** {', '.join(model_artifacts['class_names'])}")
        if 'f1_score' in model_artifacts:
            st.write(f"**F1-Score:** {model_artifacts['f1_score']:.4f}")
        if 'precision' in model_artifacts:
            st.write(f"**Precision:** {model_artifacts['precision']:.4f}")
        if 'recall' in model_artifacts:
            st.write(f"**Recall:** {model_artifacts['recall']:.4f}")

# ---- Upload Mode Selection ----
st.header("Step 1: Upload Files")
upload_mode = st.radio(
    "Select upload mode:",
    options=["Single File Pair", "Batch Folder Upload"],
    horizontal=True
)

file_pairs = []

if upload_mode == "Single File Pair":
    st.markdown("Upload a single .dat and .txt file pair with matching filenames")
    
    col1, col2 = st.columns(2)
    with col1:
        dat_file = st.file_uploader("Upload .dat file", type="dat", key="single_dat")
    with col2:
        txt_file = st.file_uploader("Upload .txt file", type="txt", key="single_txt")
    
    if dat_file and txt_file:
        dat_name = os.path.splitext(dat_file.name)[0]
        txt_name = os.path.splitext(txt_file.name)[0]
        if dat_name != txt_name:
            st.error("❌ Filenames must match except for extension!")
        else:
            tmp_dir = tempfile.mkdtemp()
            dat_path = os.path.join(tmp_dir, dat_file.name)
            txt_path = os.path.join(tmp_dir, txt_file.name)
            with open(dat_path, "wb") as f:
                f.write(dat_file.read())
            with open(txt_path, "wb") as f:
                f.write(txt_file.read())
            base_path = os.path.splitext(dat_path)[0]
            file_pairs = [(base_path, dat_name)]
            st.success(f"✅ Files uploaded successfully: `{dat_name}`")

else:  # Batch Folder Upload
    st.markdown("Upload a ZIP file containing multiple .dat and .txt file pairs")
    
    zip_file = st.file_uploader("Upload ZIP folder", type="zip", key="batch_zip")
    
    if zip_file:
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        
        # Extract ZIP
        extract_dir = os.path.join(tmp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find matching pairs
        dat_files = list(Path(extract_dir).rglob("*.dat"))
        for dat_file in dat_files:
            if dat_file.name.startswith('._'):
                continue
            txt_file = dat_file.with_suffix(".txt")
            if txt_file.exists():
                base_path = str(dat_file.with_suffix(""))
                file_name = dat_file.stem
                file_pairs.append((base_path, file_name))
        
        if file_pairs:
            st.success(f"✅ Found {len(file_pairs)} matching file pairs")
            with st.expander("View detected files"):
                for _, name in file_pairs:
                    st.write(f"• {name}")
        else:
            st.error("❌ No matching .dat/.txt file pairs found in the ZIP")

# ---- Process Files ----
if file_pairs and st.button("🔍 Run Oil Detection", type="primary"):
    st.session_state.oil_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (base_path, file_name) in enumerate(file_pairs):
        status_text.text(f"Processing {idx+1}/{len(file_pairs)}: {file_name}")
        progress_bar.progress((idx + 1) / len(file_pairs))
        
        result = process_single_file(base_path, component_type, model_artifacts, idx, len(file_pairs))
        result['file_name'] = file_name
        result['component_type'] = component_type
        st.session_state.oil_results.append(result)
    
    status_text.text(f"✅ Completed processing {len(file_pairs)} files!")
    progress_bar.empty()

# ---- Display Results ----
if st.session_state.oil_results:
    st.header("Step 2: Detection Results")
    
    # Summary statistics
    successful = sum(1 for r in st.session_state.oil_results if r['success'])
    failed = len(st.session_state.oil_results) - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(st.session_state.oil_results))
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", failed)
    
    # Display results for each file
    for idx, result in enumerate(st.session_state.oil_results):
        with st.expander(f"📄 {result['file_name']} - {result['component_type']}", expanded=(len(st.session_state.oil_results) == 1)):
            if result['success']:
                st.metric("Oil Coverage", f"{result['oil_percentage']:.2f}%")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**RGB Composite**")
                    fig1, ax1 = plt.subplots(figsize=(8, 7))
                    ax1.imshow(result['rgb_img'])
                    ax1.set_title("RGB Composite", fontsize=14, fontweight='bold')
                    ax1.axis('off')
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    st.markdown("**Oil Detection Overlay**")
                    fig2, ax2 = plt.subplots(figsize=(8, 7))
                    ax2.imshow(result['rgb_img'])
                    ax2.imshow(result['oil_mask'], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                    ax2.set_title("Oil Detection", fontsize=14, fontweight='bold')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close()
                
                # Probability map if available
                if result['probability_map'] is not None:
                    st.markdown("**Probability Map**")
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    im = ax3.imshow(result['probability_map'], cmap='hot', vmin=0, vmax=1)
                    ax3.set_title("Oil Probability Map", fontsize=14, fontweight='bold')
                    ax3.axis('off')
                    plt.colorbar(im, ax=ax3, label='Probability')
                    st.pyplot(fig3)
                    plt.close()
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # ---- Export Options ----
    st.header("Step 3: Export Results")
    
    if successful > 0:
        # Generate individual PDFs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(st.session_state.oil_results) == 1:
            result = st.session_state.oil_results[0]
            if result['success']:
                pdf_filename = f"{result['file_name']}_{result['component_type']}_report_{timestamp}.pdf"
                pdf_path = generate_pdf_report(
                    result['rgb_img'],
                    result['oil_mask'],
                    result['wavelengths'],
                    result['red_idx'],
                    result['green_idx'],
                    result['blue_idx'],
                    result['oil_percentage'],
                    pdf_filename,
                    model_name=model_artifacts['model_name'] if model_artifacts else None,
                    probability_map=result['probability_map'],
                    component_type=result['component_type']
                )
                
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
        else:
            # Batch download - create ZIP of all PDFs
            st.markdown("**Download all reports as ZIP**")
            
            if st.button("Generate All PDF Reports"):
                with st.spinner("Generating PDF reports..."):
                    zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                    
                    with zipfile.ZipFile(zip_buffer.name, 'w') as zipf:
                        for result in st.session_state.oil_results:
                            if result['success']:
                                pdf_filename = f"{result['file_name']}_{result['component_type']}_report.pdf"
                                pdf_path = generate_pdf_report(
                                    result['rgb_img'],
                                    result['oil_mask'],
                                    result['wavelengths'],
                                    result['red_idx'],
                                    result['green_idx'],
                                    result['blue_idx'],
                                    result['oil_percentage'],
                                    pdf_filename,
                                    model_name=model_artifacts['model_name'] if model_artifacts else None,
                                    probability_map=result['probability_map'],
                                    component_type=result['component_type']
                                )
                                zipf.write(pdf_path, pdf_filename)
                    
                    with open(zip_buffer.name, 'rb') as f:
                        zip_data = f.read()
                    
                    st.download_button(
                        label="📥 Download All Reports (ZIP)",
                        data=zip_data,
                        file_name=f"oil_detection_reports_{component_type}_{timestamp}.zip",
                        mime="application/zip"
                    )

# --- End of App ---
