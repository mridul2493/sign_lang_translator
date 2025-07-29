import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open("sign_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'sign_model.pkl' not found. Please make sure it's in the same directory as this script.")
        st.stop()

# Initialize MediaPipe
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, static_image_mode=False, min_detection_confidence=0.5)
    return hands

def predict_from_frame(image, model, hands, prev_letter, stable_count, stable_threshold=15):
    """
    Predict sign language letter from a frame
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_letter = ""
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            data = np.array(data).reshape(1, -1)

            prediction = model.predict(data)[0]
            current_letter = prediction

            if prediction == prev_letter:
                stable_count += 1
            else:
                stable_count = 0
            prev_letter = prediction

    return current_letter, prev_letter, stable_count

# Initialize Streamlit app
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👋",
    layout="wide"
)

st.title("👋 Sign Language Recognition App")
st.write("This app recognizes American Sign Language (ASL) letters from your webcam feed.")

# Initialize session state variables
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'prev_letter' not in st.session_state:
    st.session_state.prev_letter = ""
if 'stable_count' not in st.session_state:
    st.session_state.stable_count = 0
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False

# Load model and initialize MediaPipe
model = load_model()
hands = init_mediapipe()

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Camera Feed")
    
    # Camera input
    camera_input = st.camera_input("Take a picture to recognize sign language", key="camera")
    
    if camera_input is not None:
        # Convert the uploaded image to OpenCV format
        image = Image.open(camera_input)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Make prediction
        current_letter, st.session_state.prev_letter, st.session_state.stable_count = predict_from_frame(
            frame, model, hands, st.session_state.prev_letter, st.session_state.stable_count
        )
        
        # Add letter to word if stable
        stable_threshold = 15
        if (current_letter == st.session_state.prev_letter and 
            st.session_state.stable_count >= stable_threshold and 
            current_letter != ""):
            st.session_state.current_word += current_letter
            st.session_state.stable_count = 0
        
        # Display the image
        st.image(image, caption="Current Frame", use_column_width=True)

with col2:
    st.subheader("🔤 Recognition Results")
    
    # Current letter
    if 'current_letter' in locals():
        st.metric("Current Letter", current_letter if current_letter else "None")
    else:
        st.metric("Current Letter", "None")
    
    # Current word
    st.metric("Current Word", st.session_state.current_word if st.session_state.current_word else "")
    
    # Stability indicator
    if 'current_letter' in locals() and current_letter:
        stability_percentage = min(100, (st.session_state.stable_count / 15) * 100)
        st.progress(stability_percentage / 100)
        st.caption(f"Stability: {stability_percentage:.0f}%")
    
    # Control buttons
    st.subheader("🎮 Controls")
    
    if st.button("🔄 Reset Word", type="primary"):
        st.session_state.current_word = ""
        st.session_state.prev_letter = ""
        st.session_state.stable_count = 0
        st.success("Word reset!")
        st.rerun()
    
    if st.button("⌫ Remove Last Letter"):
        if st.session_state.current_word:
            st.session_state.current_word = st.session_state.current_word[:-1]
            st.success("Last letter removed!")
            st.rerun()
    
    if st.button("📝 Add Space"):
        st.session_state.current_word += " "
        st.success("Space added!")
        st.rerun()

# Instructions
with st.expander("📋 Instructions"):
    st.write("""
    1. **Allow camera access** when prompted by your browser
    2. **Position your hand** clearly in front of the camera
    3. **Make ASL letter signs** - hold each sign steady for recognition
    4. **Watch the stability meter** - letters are added when the meter fills up
    5. **Use the controls** to reset words, remove letters, or add spaces
    
    **Tips:**
    - Ensure good lighting for better recognition
    - Keep your hand centered in the camera view
    - Hold signs steady for a few seconds for recognition
    - The app recognizes static ASL letters (A-Z)
    """)

# Settings
with st.expander("⚙️ Settings"):
    st.write("**Model Information:**")
    st.write("- Model: Pre-trained ASL letter recognition")
    st.write("- Max hands detected: 2")
    st.write("- Stability threshold: 15 frames")
    
    if st.button("🧹 Clear All Data"):
        st.session_state.current_word = ""
        st.session_state.prev_letter = ""
        st.session_state.stable_count = 0
        st.success("All data cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Sign Language Recognition with MediaPipe")