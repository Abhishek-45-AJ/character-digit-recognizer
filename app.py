import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# =====================================================================
# STEP 1: INITIAL UI PAGE SETTINGS
# =====================================================================
st.set_page_config(page_title="AI Character Recognition", layout="centered")

# =====================================================================
# STEP 2: NATIVE STANDALONE .H5 MODEL LOADER 
# =====================================================================
@st.cache_resource
def load_h5_model():
    try:
        # Now that versions are matched, this single line will work flawlessly!
        return tf.keras.models.load_model('emnist_model.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model file: {e}. Check if 'emnist_model.h5' is inside this folder.")
        return None

model = load_h5_model()

# =====================================================================
# STEP 3: THE 62 CLASS TRANSLATION MAP
# =====================================================================
DIGITS = list("0123456789")
UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
MASTER_MAP = DIGITS + UPPERCASE + LOWERCASE

# =====================================================================
# STEP 4: INTERACTIVE FRONTEND UI LAYOUT
# =====================================================================
st.title("Smart Character Recognition")
st.write("Draw clearly inside the black box, then click the **Run Recognition** button.")

col1, col2 = st.columns(2)

# --- LEFT PANEL: DRAWING SPACE ---
with col1:
    st.markdown("### 👇 Draw Here")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=24,                     
        stroke_color="#FFFFFF",              
        background_color="#000000",          
        height=280,                          
        width=280,                           
        drawing_mode="freedraw",
        update_streamlit=True,               
        key="canvas",
    )
    predict_button = st.button("🚀 Run Recognition", type="primary")

# --- RIGHT PANEL: AI RESULTS ---
with col2:
    st.markdown("### 🤖 AI Prediction")
    
    if predict_button:
        if canvas_result.image_data is not None:
            raw_pixels = canvas_result.image_data
            img = Image.fromarray(raw_pixels.astype('uint8')).convert('L')
            
            if np.max(np.array(img)) > 0:
                # Bounding-box auto-centering layout
                bbox = img.getbbox()
                if bbox:
                    img_cropped = img.crop(bbox)
                    img_padded = ImageOps.expand(img_cropped, border=30, fill=0)
                else:
                    img_padded = img
                
                img_resized = img_padded.resize((28, 28))
                img_normalized = np.array(img_resized).astype(np.float32) / 255.0
                img_input = img_normalized.reshape(1, 28, 28, 1)
                
                if model is not None:
                    prediction_probabilities = model.predict(img_input)
                    best_match_index = np.argmax(prediction_probabilities)
                    accuracy_score = np.max(prediction_probabilities) * 100
                    
                    final_character = MASTER_MAP[best_match_index]
                    
                    st.success(f"Identified Character: **{final_character}**")
                    st.metric(label="Prediction Confidence Score", value=f"{accuracy_score:.2f}%")
                else:
                    st.error("Model engine failed to activate properly.")
            else:
                st.warning("The sketch box is empty! Please draw a character before predicting.")
        else:
            st.error("Canvas matrix extraction error: Unable to read image data.")
    else:
        st.info("Draw a letter or number and click 'Run Recognition' to test.")
