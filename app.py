import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct, idct

# --- Page Configuration ---
st.set_page_config(
    page_title="OptiMatrix AI | SVD vs DCT",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Styling (Nanopredict Aesthetic) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #3e4253; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { color: #00d4ff; font-family: 'Inter', sans-serif; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        background: linear-gradient(45deg, #00d4ff, #008fb3); 
        color: white; 
        font-weight: bold; 
        border: none;
    }
    .stDownloadButton>button {
        background: transparent;
        border: 1px solid #00d4ff;
        color: #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic Functions ---

def apply_svd(matrix, k):
    """Decomposes matrix into U, S, V and reconstructs using k components."""
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    reconstructed = (u[:, :k] @ np.diag(s[:k]) @ vh[:k, :])
    return reconstructed, s

def apply_dct(matrix, quality):
    """Applies 2D Discrete Cosine Transform and truncates high frequencies."""
    coeff = dct(dct(matrix.T, norm='ortho').T, norm='ortho')
    row, col = coeff.shape
    keep = quality / 100.0
    coeff[int(row*keep):, :] = 0
    coeff[:, int(col*keep):] = 0
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

def get_metrics(original, compressed):
    """Calculates PSNR and SSIM to evaluate compression quality."""
    p_val = psnr(original, compressed, data_range=255)
    # SSIM windowing for RGB vs Grayscale
    win_size = 3 if original.ndim == 3 else 7
    s_val = ssim(original, compressed, data_range=255, 
                 channel_axis=-1 if original.ndim == 3 else None, 
                 win_size=win_size)
    return p_val, s_val

def get_storage_info(h, w, k, channels):
    """Calculates memory footprint using SVD rank reduction."""
    original_elements = h * w * channels
    compressed_elements = (h * k + k + w * k) * channels
    savings = (1 - (compressed_elements / original_elements)) * 100
    return savings

# --- UI Header ---
st.title("💠 OptiMatrix AI: Image Compression Suite")
st.markdown("### Comparing SVD & DCT Architectures via Linear Algebra")
st.divider()

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📤 Input & Controls")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        c = 1 if len(img_array.shape) == 2 else 3
        
        st.info(f"Resolution: {w}x{h} | Channels: {c}")
        
        st.subheader("Compression Strength")
        k_val = st.slider("SVD Rank (k)", 1, min(h, w), int(min(h, w)*0.1))
        q_val = st.slider("DCT Quality %", 1, 100, 20)
        
        st.divider()
        st.subheader("Visuals")
        show_anim = st.checkbox("Rank-by-Rank Animation")
        show_plot = st.checkbox("Show Singular Value Decay", value=True)

# --- Main Dashboard ---
if uploaded_file:
    # 1. Processing (Handles RGB and Grayscale)
    if c == 1:
        svd_raw, s_vals = apply_svd(img_array, k_val)
        dct_raw = apply_dct(img_array, q_val)
    else:
        svd_raw, dct_raw = np.zeros_like(img_array, dtype=float), np.zeros_like(img_array, dtype=float)
        s_vals_list = []
        for i in range(3):
            res, s = apply_svd(img_array[:,:,i], k_val)
            svd_raw[:,:,i] = res
            s_vals_list.append(s)
            dct_raw[:,:,i] = apply_dct(img_array[:,:,i], q_val)
        s_vals = np.mean(s_vals_list, axis=0)

    svd_res = np.clip(svd_raw, 0, 255).astype(np.uint8)
    dct_res = np.clip(dct_raw, 0, 255).astype(np.uint8)

    # 2. Top Metric Row
    p_svd, s_svd = get_metrics(img_array, svd_res)
    p_dct, s_dct = get_metrics(img_array, dct_res)
    savings = get_storage_info(h, w, k_val, c)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Memory Saved (SVD)", f"{savings:.1f}%")
    m2.metric("SVD Quality (PSNR)", f"{p_svd:.2f} dB")
    m3.metric("DCT Quality (PSNR)", f"{p_dct:.2f} dB")
    m4.metric("Structural Similarity", f"{s_svd:.3f}")

    # 3. Image Comparison Display
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Original")
        st.image(img, use_container_width=True)
        st.caption("Raw Image Data")

    with col2:
        st.markdown("#### SVD Output")
        st.image(svd_res, use_container_width=True)
        
        # --- FIXED BLOCK ---
        buf = io.BytesIO()
        # Convert to RGB to ensure JPEG compatibility
        svd_img = Image.fromarray(svd_res)
        if svd_img.mode != 'RGB':
            svd_img = svd_img.convert('RGB')
        svd_img.save(buf, format="JPEG")
        # -------------------
        
        st.download_button("Download SVD Result", buf.getvalue(), "svd_compressed.jpg")

    with col3:
        st.markdown("#### DCT Output")
        st.image(dct_res, use_container_width=True)
        
        # --- FIXED BLOCK ---
        buf2 = io.BytesIO()
        dct_img = Image.fromarray(dct_res)
        if dct_img.mode != 'RGB':
            dct_img = dct_img.convert('RGB')
        dct_img.save(buf2, format="JPEG")
        # -------------------
        
        st.download_button("Download DCT Result", buf2.getvalue(), "dct_compressed.jpg")

    # 4. Feature: Animation
    if show_anim:
        st.divider()
        st.subheader("🎬 Rank Reconstruction Animation")
        anim_placeholder = st.empty()
        rank_label = st.empty()
        
        # Determine frame steps based on k_val to keep animation smooth but fast
        num_frames = 20
        frames = np.unique(np.linspace(1, k_val, num_frames, dtype=int))
        
        for step in frames:
            if c == 1:
                f, _ = apply_svd(img_array, step)
            else:
                f = np.zeros_like(img_array, dtype=float)
                for i in range(3):
                    res, _ = apply_svd(img_array[:,:,i], step)
                    f[:,:,i] = res
            
            f_display = np.clip(f, 0, 255).astype(np.uint8)
            anim_placeholder.image(f_display, width=600)
            rank_label.markdown(f"**Current Approximation Rank: {step}**")
            time.sleep(0.1)
        st.success("Reconstruction Loop Complete!")

    # 5. Feature: SVD Math Visualization
    if show_plot:
        st.divider()
        st.subheader("📈 Energy Distribution (Singular Values)")
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.style.use('dark_background')
        ax.plot(s_vals[:200], color='#00d4ff', linewidth=2)
        ax.fill_between(range(len(s_vals[:200])), s_vals[:200], color='#00d4ff', alpha=0.1)
        ax.axvline(x=k_val, color='red', linestyle='--', label=f'Current k={k_val}')
        ax.set_title("Information Decay across Singular Values")
        ax.set_ylabel("Magnitude (Log Scale)")
        ax.set_yscale('log')
        ax.legend()
        st.pyplot(fig)

    # 6. Comparison Table
    st.divider()
    st.subheader("🔎 Architectural Comparison Report")
    comparison_data = {
        "Criteria": ["Math Principle", "Efficiency", "Artifacts", "Best For"],
        "SVD (Algebraic)": ["Singular Value Decomposition", f"High ({savings:.1f}% saved)", "Ghosting/Blurring", "Noise Reduction"],
        "DCT (Frequency)": ["Cosine Transform", f"Variable ({q_val}% quality)", "Blocking/Pixelation", "Web Standards (JPEG)"]
    }
    st.table(comparison_data)

else:
    st.warning("👈 Please upload an image in the sidebar to initialize the OptiMatrix suite.")