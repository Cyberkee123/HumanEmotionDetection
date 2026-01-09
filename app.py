import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# 1. Initialize session state variables at the top
if 'prediction_label' not in st.session_state:
   st.session_state.prediction_label = None
if 'confidence_score' not in st.session_state:
   st.session_state.confidence_score = None

# -------------------------------
# Load VGG19 Transfer Learning Model
# -------------------------------
@st.cache_resource
def load_emotion_model():
    # Ensure this filename matches the one you save in your Jupyter Notebook
    model = tf.keras.models.load_model(
        "full_emotion_model.keras",
        compile=False
    )
    return model

model = load_emotion_model()

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Human Emotion Detection (VGG19 Transfer Learning)")
st.write("Upload an image for high-accuracy emotion prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Convert the file to an OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # 3. Preprocess for VGG19 (RGB + Resize to 48x48)
    # VGG19 requires 3 channels (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (48, 48))
    
    # 4. Normalize and Reshape
    # Convert to float32, scale to [0, 1]
    img_array = resized_image.astype('float32') / 255.0
    # Add batch dimension: (1, 48, 48, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prediction
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing facial expression with VGG19...'):
            predictions = model.predict(img_array)
            max_index = np.argmax(predictions[0])
            
            # Save to session state to prevent disappearing on rerun
            st.session_state.prediction_label = emotion_labels[max_index]
            st.session_state.confidence_score = predictions[0][max_index] * 100

# 6. Display Result (Outside the 'if uploaded_file' block to persist)
if st.session_state.prediction_label:
   st.success(f"Result: {st.session_state.prediction_label.upper()}")
   st.info(f"Confidence: {st.session_state.confidence_score:.2f}%")