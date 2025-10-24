import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from PIL import Image
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import joblib

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
def load_model():
    """Load the trained model from the same directory"""
    model_path = 'oil_detection_model.pkl'
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file '{model_path}' not found in the current directory.")
        st.info("The app will run in placeholder mode. To use a trained model, place 'oil_detection_model.pkl' in the same directory as this script.")
        return None
    
    try:
        artifacts = joblib.load(model_path)
        st.success(f"✅ Model loaded: {artifacts['model_name']}")
        return artifacts
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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
    background = model_artifacts['background']
    scaler = model_artifacts['scaler']
    oil_label_index = model_artifacts['oil_label_index']
    
    # Get shape
    B, H, W = rad_data.shape
    
    # Reshape data: (B, H, W) -> (H*W, B)

    rad_data = rad_data / background['radiance']
    rad_data = np.nan_to_num(rad_data, nan=1.0, posinf=1.0, neginf=1.0)

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


def generate_pdf_report(rgb_img, oil_mask, wavelengths, red_idx, green_idx, blue_idx, 
                       oil_percentage, filename, model_name=None, probability_map=None):
    """Generate a PDF report with analysis results"""
    
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Hyperspectral Oil Detection Report', fontsize=20, fontweight='bold', y=0.95)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.text(0.5, 0.85, f'Generated: {timestamp}', ha='center', fontsize=12)
        
        # Summary statistics
        model_info = f"\nModel Used: {model_name}" if model_name else "\nModel Used: Placeholder (Random)"
        summary_text = f"""
        Analysis Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {model_info}
        
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
        
        plt.text(0.1, 0.65, summary_text, fontsize=11, verticalalignment='top', 
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


# ============================================
# Streamlit App
# ============================================

# ---- App Title ----
st.set_page_config(page_title="HSC Oil Detection", layout="centered")
st.title("🛢️ Hyperspectral Oil Detection Tool")

st.write("""
This Streamlit app lets you upload a pair of hyperspectral files (.dat + .txt), visualize an RGB composite,
and run an oil detection algorithm using a trained machine learning model.
""")

# Initialize session state to preserve results
if 'oil_result' not in st.session_state:
    st.session_state.oil_result = None
if 'rgb_img' not in st.session_state:
    st.session_state.rgb_img = None
if 'oil_percentage' not in st.session_state:
    st.session_state.oil_percentage = 0.0
if 'red_idx' not in st.session_state:
    st.session_state.red_idx = 0
if 'green_idx' not in st.session_state:
    st.session_state.green_idx = 0
if 'blue_idx' not in st.session_state:
    st.session_state.blue_idx = 0
if 'probability_map' not in st.session_state:
    st.session_state.probability_map = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# ---- Load Model on Startup ----
with st.spinner("Loading model..."):
    model_artifacts = load_model()

if model_artifacts is not None:
    st.session_state.model_name = model_artifacts['model_name']
    
    # Display model info in sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model:** {model_artifacts['model_name']}")
        st.write(f"**Classes:** {', '.join(model_artifacts['class_names'])}")
        st.write(f"**F1-Score:** {model_artifacts.get('f1_score', 'N/A'):.4f}" if 'f1_score' in model_artifacts else "")
        st.write(f"**Precision:** {model_artifacts.get('precision', 'N/A'):.4f}" if 'precision' in model_artifacts else "")
        st.write(f"**Recall:** {model_artifacts.get('recall', 'N/A'):.4f}" if 'recall' in model_artifacts else "")

# ---- Upload Section ----

st.header("Step 1: Upload HSC Files")
st.markdown("Upload both files with matching filenames (e.g., `sample.dat` and `sample.txt`)")

dat_file = st.file_uploader("Upload .dat file", type="dat")
txt_file = st.file_uploader("Upload corresponding .txt file", type="txt")

file_ready = False
base_path = None
rad_data = None
wavelengths = None

if dat_file and txt_file:
    dat_name = os.path.splitext(dat_file.name)[0]
    txt_name = os.path.splitext(txt_file.name)[0]
    if dat_name != txt_name:
        st.error("❌ Filenames must match except for extension!")
    else:
        # Save uploaded files to temporary location
        tmp_dir = tempfile.mkdtemp()
        dat_path = os.path.join(tmp_dir, dat_file.name)
        txt_path = os.path.join(tmp_dir, txt_file.name)
        with open(dat_path, "wb") as f: 
            f.write(dat_file.read())
        with open(txt_path, "wb") as f: 
            f.write(txt_file.read())
        base_path = os.path.splitext(dat_path)[0]
        file_ready = True
        st.success(f"✅ Files uploaded successfully: `{dat_name}`")

# ---- Data Conversion ----

if file_ready:
    st.subheader("Data Conversion")
    with st.spinner("Reading and converting uploaded data..."):
        try:
            # --- Insert conversion code snippet here ---
            # Call the actual HSC data reading function
            rad_data, wavelengths = read_hsc_data(base_path)
            
            st.success(f"✅ Data loaded successfully!")
            st.write(f"**Radiance Data Shape:** {rad_data.shape} (Bands × Height × Width)")
            st.write(f"**Data Type:** {rad_data.dtype}")
            st.write(f"**Number of Bands:** {rad_data.shape[0]}")
            st.write(f"**Wavelength Range:** {wavelengths[0]:.1f}nm - {wavelengths[-1]:.1f}nm")
                
        except Exception as e:
            st.error(f"❌ Error reading data: {str(e)}")
            st.exception(e)

# ---- RGB Composite Generation ----

rgb_img = None

if rad_data is not None and wavelengths is not None:
    st.header("Step 2: RGB Composite Visualization")
    st.markdown("Select wavelengths for each RGB channel from the available bands.")
    
    # Determine default wavelengths based on available data
    def find_closest_wavelength(target, available_wls):
        """Find the closest available wavelength to the target"""
        idx = np.argmin(np.abs(available_wls - target))
        return float(available_wls[idx])
    
    # Set intelligent defaults based on available wavelengths
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
    
    # Display available wavelengths for selection
    st.info(f"Available wavelengths: {', '.join([f'{w:.1f}nm' for w in wavelengths])}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        red_wl = st.selectbox(
            "🔴 Red wavelength (nm)", 
            options=wavelengths,
            index=int(np.argmin(np.abs(wavelengths - default_red_wl))),
            format_func=lambda x: f"{x:.1f} nm"
        )
    with col2:
        green_wl = st.selectbox(
            "🟢 Green wavelength (nm)", 
            options=wavelengths,
            index=int(np.argmin(np.abs(wavelengths - default_green_wl))),
            format_func=lambda x: f"{x:.1f} nm"
        )
    with col3:
        blue_wl = st.selectbox(
            "🔵 Blue wavelength (nm)", 
            options=wavelengths,
            index=int(np.argmin(np.abs(wavelengths - default_blue_wl))),
            format_func=lambda x: f"{x:.1f} nm"
        )
    
    # --- Insert RGB composite generation snippet here ---
    # Find exact wavelength indices
    red_idx = int(np.where(wavelengths == red_wl)[0][0])
    green_idx = int(np.where(wavelengths == green_wl)[0][0])
    blue_idx = int(np.where(wavelengths == blue_wl)[0][0])
    
    # Store in session state
    st.session_state.red_idx = red_idx
    st.session_state.green_idx = green_idx
    st.session_state.blue_idx = blue_idx
    
    # Extract channels
    red_ch = rad_data[red_idx]
    green_ch = rad_data[green_idx]
    blue_ch = rad_data[blue_idx]
    
    # Normalize function
    def normalize(channel):
        min_val, max_val = np.min(channel), np.max(channel)
        return (channel - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(channel)
    
    # Create RGB composite
    rgb_img = np.stack([
        normalize(red_ch), 
        normalize(green_ch), 
        normalize(blue_ch)
    ], axis=2)
    
    # Store in session state
    st.session_state.rgb_img = rgb_img
    
    st.info(f"Using bands: R={red_idx} ({wavelengths[red_idx]:.1f}nm), "
            f"G={green_idx} ({wavelengths[green_idx]:.1f}nm), "
            f"B={blue_idx} ({wavelengths[blue_idx]:.1f}nm)")
    
    # Display RGB composite
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb_img)
    ax.set_title("RGB Composite Image", fontsize=16, fontweight='bold')
    ax.axis('off')
    st.pyplot(fig)
    plt.close()

# ---- Oil Detection Algorithm ----

if st.session_state.rgb_img is not None:
    st.header("Step 3: Oil Detection")
    
    if model_artifacts is not None:
        st.markdown(f"""
        Click the button below to run oil detection using the **{model_artifacts['model_name']}** model.
        """)
    else:
        st.markdown("""
        Click the button below to run oil detection analysis on the hyperspectral data.
        
        **Note:** No model found. Using placeholder detection. Place 'oil_detection_model.pkl' in the same directory for trained model predictions.
        """)
    
    if st.button("🔍 Run Oil Detection", type="primary"):
        with st.spinner("Running oil detection algorithm..."):
            # --- Insert oil detection algorithm here ---
            
            if model_artifacts is not None and rad_data is not None:
                # Use actual trained model
                try:
                    oil_mask, probability_map = predict_oil_pixels(rad_data, wavelengths, model_artifacts)
                    st.session_state.oil_result = oil_mask
                    st.session_state.probability_map = probability_map
                    st.session_state.oil_percentage = (oil_mask.sum() / oil_mask.size) * 100
                    st.success(f"✅ Oil detection completed using {model_artifacts['model_name']}!")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
            else:
                # Placeholder implementation
                oil_mask = np.random.rand(*st.session_state.rgb_img.shape[:2])
                oil_mask = (oil_mask > 0.85).astype(np.float32)
                
                st.session_state.oil_result = oil_mask
                st.session_state.probability_map = None
                st.session_state.oil_percentage = (oil_mask.sum() / oil_mask.size) * 100
                st.success("✅ Oil detection completed (placeholder mode)!")
    
    # Display results if they exist
    if st.session_state.oil_result is not None:
        st.subheader("Detection Results")
        
        # Statistics
        st.metric("Oil Coverage", f"{st.session_state.oil_percentage:.2f}%")
        
        # Display images stacked vertically for better visibility
        st.markdown("### Original RGB Composite")
        fig1, ax1 = plt.subplots(figsize=(14, 12))
        ax1.imshow(st.session_state.rgb_img)
        ax1.set_title("Original RGB Composite", fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close()
        
        st.markdown("### Oil Detection Overlay")
        fig2, ax2 = plt.subplots(figsize=(14, 12))
        ax2.imshow(st.session_state.rgb_img)
        ax2.imshow(st.session_state.oil_result, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        ax2.set_title("Oil Region Overlay (Red)", fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close()
        
        # Show probability map if available
        if st.session_state.probability_map is not None:
            st.markdown("### Oil Probability Map")
            fig3, ax3 = plt.subplots(figsize=(14, 12))
            im = ax3.imshow(st.session_state.probability_map, cmap='hot', vmin=0, vmax=1)
            ax3.set_title("Oil Probability Map", fontsize=16, fontweight='bold', pad=20)
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, label='Probability', fraction=0.046, pad=0.04)
            st.pyplot(fig3)
            plt.close()
        
        if model_artifacts is None:
            st.info("🔴 Red overlay shows detected oil regions. Place 'oil_detection_model.pkl' in the app directory for actual trained model predictions.")

# ---- Save/Export Options ----
if st.session_state.oil_result is not None and wavelengths is not None:
    st.header("Step 4: Export Results")
    st.markdown("Download a comprehensive PDF report with all analysis results.")
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"oil_detection_report_{timestamp}.pdf"
    
    # Generate PDF without button click
    pdf_path = generate_pdf_report(
        st.session_state.rgb_img, 
        st.session_state.oil_result, 
        wavelengths, 
        st.session_state.red_idx, 
        st.session_state.green_idx, 
        st.session_state.blue_idx,
        st.session_state.oil_percentage,
        pdf_filename,
        model_name=st.session_state.model_name,
        probability_map=st.session_state.probability_map
    )
    
    # Read PDF file
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    # Provide download button
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_data,
        file_name=pdf_filename,
        mime="application/pdf",
        type="primary"
    )

# --- End of App ---