import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# Premium Custom CSS Injection for Outstanding Aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
    }
    
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem !important;
        padding-bottom: 5px !important;
        text-shadow: 0 4px 20px rgba(56, 189, 248, 0.15) !important;
    }
    
    .section-header {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2) !important;
        padding-bottom: 8px !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 25px !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    
    .card:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 2px dashed rgba(56, 189, 248, 0.3) !important;
        border-radius: 12px !important;
        padding: 25px !important;
        transition: border-color 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.6) !important;
    }
    
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(8px) !important;
        padding: 20px !important;
    }
    
    .stSuccess {
        border-left: 5px solid #38bdf8 !important; /* Unified Theme Accent Color */
    }
    .stError {
        border-left: 5px solid #ef4444 !important;
    }
    .stWarning {
        border-left: 5px solid #f59e0b !important;
    }
    .stInfo {
        border-left: 5px solid #3b82f6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# MODEL LOADING
# -------------------

def create_model():
    # Programmatic reconstruction of the exact CNN model architecture from training
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(3, activation='softmax', name='dense_1')
    ])
    return model

@st.cache_resource
def load():
    model = create_model()
    
    # Load weights directly to bypass the version-mismatched Keras JSON deserializer
    loaded = False
    for name in ["road_damage_cnn_model.keras", "road_damage_cnn_model.h5"]:
        if os.path.exists(name):
            try:
                model.load_weights(name)
                loaded = True
                break
            except Exception as e:
                continue
                
    if not loaded:
        raise FileNotFoundError("Could not load weights from road_damage_cnn_model.keras or road_damage_cnn_model.h5.")
        
    return model

try:
    model = load()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class mappings matching the Keras model output layers
labels = {
    "0": "Pothole Detected",
    "1": "Crack Detected",
    "2": "Manhole Detected"
}

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    # Convert RGBA to RGB if needed
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

# =====================================================================
# SECTION 1 — Header
# =====================================================================
st.title("AI-Based Road Damage Detection System")
st.caption("Smart City Infrastructure Monitoring using CNN")
st.divider()

# =====================================================================
# SECTION 2 — About the Project
# =====================================================================
st.markdown('<h2 class="section-header">SECTION 2 — About the Project</h2>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
        <h3 style="color: #38bdf8; margin-top: 0;">Project Context & Architecture</h3>
        <p><b>Why Road Monitoring is Important:</b> Maintaining high-quality road conditions is critical for safety and efficiency. Regular monitoring reduces traffic accidents, prevents vehicle wear and tear, minimizes repair expenses, and ensures the long-term sustainability of municipal infrastructure.</p>
        <p><b>Role of CNN in Computer Vision:</b> Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. CNNs automatically extract spatial hierarchies of features from raw images—going from basic edges to complex textures and objects. This process is shift-invariant and allows high-accuracy classification under varying illumination and weather conditions.</p>
        <p><b>Practical Industry Applications:</b>
            <ul style="margin-top: 0; padding-left: 20px;">
                <li><b>Municipal Maintenance Dispatch:</b> Automating work orders for repair crews</li>
                <li><b>Autonomous Vehicles:</b> Enabling real-time road obstacle and damage avoidance</li>
                <li><b>Infrastructure Asset Management:</b> Performing large-scale highway health assessment from dashcam video feeds</li>
            </ul>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================================
# SECTION 3 — Upload Area
# =====================================================================
st.markdown('<h2 class="section-header">SECTION 3 — Upload Area</h2>', unsafe_allow_html=True)
file = st.file_uploader(
    "Drag and drop or browse a road surface image (PNG, JPG, JPEG):",
    type=["jpg", "png", "jpeg"]
)
st.divider()

if file:
    image = Image.open(file)
    col1, col2 = st.columns(2)

    with col1:
        # =====================================================================
        # SECTION 4 — Uploaded Image Preview
        # =====================================================================
        st.markdown('<h2 class="section-header">SECTION 4 — Image Preview</h2>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded Road Image")

    x = preprocess(image)
    pred = model.predict(x, verbose=0)
    cls = np.argmax(pred)
    conf = np.max(pred) * 100
    label = labels[str(cls)]

    if conf > 85:
        severity = "High"
    elif conf > 60:
        severity = "Medium"
    else:
        severity = "Low"

    with col2:
        # =====================================================================
        # SECTION 5 — Prediction Area
        # =====================================================================
        st.markdown('<h2 class="section-header">SECTION 5 — Prediction Area</h2>', unsafe_allow_html=True)
        
        st.success(
            f"🎯 **Prediction:** {label}  \n"
            f"📊 **Confidence:** {conf:.2f}%  \n"
            f"⚠️ **Severity Level:** {severity}"
        )

        # =====================================================================
        # SECTION 7 — Recommendations
        # =====================================================================
        st.markdown('<h2 class="section-header">SECTION 7 — Recommendations</h2>', unsafe_allow_html=True)
        
        if severity == "High":
            st.error(
                "Immediate maintenance recommended.  \n"
                "High-risk road condition detected."
            )
        elif severity == "Medium":
            st.warning(
                "Scheduled repair dispatch recommended.  \n"
                "Medium-risk road condition detected."
            )
        else:
            st.info(
                "Normal monitoring schedules recommended.  \n"
                "Low-risk road condition detected."
            )

    st.divider()

    # =====================================================================
    # SECTION 6 — Visualization Area
    # =====================================================================
    st.markdown('<h2 class="section-header">SECTION 6 — Visualization Area</h2>', unsafe_allow_html=True)
    st.subheader("Class Confidence Probability Chart")
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('none')  # Transparent background for seamless look
    ax.set_facecolor('none')
    
    # Unified bar color theme (#38bdf8) throughout the entire chart
    theme_color = '#38bdf8'
    bars = ax.bar(
        [name.replace(" Detected", "") for name in labels.values()],
        pred[0] * 100,
        color=theme_color,
        edgecolor='none',
        width=0.4
    )
    
    # Custom spines and grid
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    ax.set_ylabel('Confidence (%)', color='#cbd5e1', fontsize=12)
    ax.set_ylim(0, 110)
    
    # Value annotations on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='#f8fafc', fontweight='bold', fontsize=10)
                    
    st.pyplot(fig)
else:
    st.info("💡 **Awaiting Road Image Upload:** Please select or drag a road image to activate the model predictions.")