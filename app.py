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
        border-left: 5px solid #38bdf8 !important;
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
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(285px, 1fr)); gap: 20px;">
            <div>
                <h4 style="color: #38bdf8; margin: 0 0 8px 0; font-size: 1.1rem;">🛡️ Importance</h4>
                <p style="font-size: 0.95rem; margin: 0; color: #cbd5e1; line-height: 1.45;">Critical for public safety. Reduces severe vehicle accidents, limits tyre damage, and reduces high municipal repair costs.</p>
            </div>
            <div>
                <h4 style="color: #818cf8; margin: 0 0 8px 0; font-size: 1.1rem;">🧠 CNN Classification</h4>
                <p style="font-size: 0.95rem; margin: 0; color: #cbd5e1; line-height: 1.45;">Automatically extracts spatial hierarchies (edges to object textures) to perform robust predictions in real-time.</p>
            </div>
            <div>
                <h4 style="color: #c084fc; margin: 0 0 8px 0; font-size: 1.1rem;">🚀 Practical Applications</h4>
                <p style="font-size: 0.95rem; margin: 0; color: #cbd5e1; line-height: 1.45;">Real-time automated city monitoring, proactive crew dispatch, and autonomous vehicle navigation safety.</p>
            </div>
        </div>
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

    # Define glowing card variables matching severity
    if severity == "High":
        glow_color = "#ef4444"
        bg_glow = "rgba(239, 68, 68, 0.08)"
        border_glow = "rgba(239, 68, 68, 0.4)"
        badge_icon = "🚨"
    elif severity == "Medium":
        glow_color = "#f59e0b"
        bg_glow = "rgba(245, 158, 11, 0.08)"
        border_glow = "rgba(245, 158, 11, 0.4)"
        badge_icon = "⚠️"
    else:
        glow_color = "#38bdf8"
        bg_glow = "rgba(56, 189, 248, 0.08)"
        border_glow = "rgba(56, 189, 248, 0.4)"
        badge_icon = "ℹ️"

    with col2:
        # =====================================================================
        # SECTION 5 — Prediction Area
        # =====================================================================
        st.markdown('<h2 class="section-header">SECTION 5 — Prediction Area</h2>', unsafe_allow_html=True)
        
        # Giant glowing result highlight card
        st.markdown(
            f"""
            <div style="
                background: {bg_glow};
                border: 2px solid {border_glow};
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 0 25px {border_glow};
                text-align: center;
                margin-bottom: 25px;
            ">
                <span style="font-size: 2.8rem;">{badge_icon}</span>
                <h3 style="color: #ffffff; margin: 10px 0 5px 0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.9;">Analysis Result</h3>
                <h1 style="
                    color: {glow_color} !important; 
                    font-size: 2.8rem !important; 
                    font-weight: 800 !important; 
                    margin: 5px 0 !important;
                    background: none !important;
                    -webkit-text-fill-color: initial !important;
                    text-shadow: 0 0 10px {border_glow} !important;
                ">
                    {label.upper()}
                </h1>
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 18px;">
                    <div style="background: rgba(255,255,255,0.04); padding: 10px 20px; border-radius: 10px; min-width: 120px;">
                        <span style="color: #cbd5e1; font-size: 0.8rem; display: block; margin-bottom: 2px;">CONFIDENCE</span>
                        <strong style="color: #ffffff; font-size: 1.3rem;">{conf:.2f}%</strong>
                    </div>
                    <div style="background: rgba(255,255,255,0.04); padding: 10px 20px; border-radius: 10px; min-width: 120px;">
                        <span style="color: #cbd5e1; font-size: 0.8rem; display: block; margin-bottom: 2px;">SEVERITY</span>
                        <strong style="color: {glow_color}; font-size: 1.3rem;">{severity.upper()}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
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
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    theme_color = '#38bdf8'
    bars = ax.bar(
        [name.replace(" Detected", "") for name in labels.values()],
        pred[0] * 100,
        color=theme_color,
        edgecolor='none',
        width=0.4
    )
    
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.tick_params(colors='#cbd5e1', labelsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.1)
    ax.set_ylabel('Confidence (%)', color='#cbd5e1', fontsize=12)
    ax.set_ylim(0, 110)
    
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