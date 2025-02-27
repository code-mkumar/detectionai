import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS  # Google Text-to-Speech
import os

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# LM Studio API URL
LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

# Streamlit UI
st.title("🚀 Object Detection & AI Voice Assistant")
st.write("Capture an image using your webcam or upload one.")

# Camera Input
captured_image = st.camera_input("📸 Take a picture using your webcam")

# Upload image
uploaded_file = st.file_uploader("📂 Or upload an image", type=["jpg", "png", "jpeg"])

# Text box for user query
#user_query = st.text_input("🔍 Enter your query:")

# Button to process query
#query_submitted = st.button("Search Answer")

# Select the image source
image_source = captured_image if captured_image else uploaded_file

# Initialize description
description = ""

if image_source:
    # Load image
    image = Image.open(image_source)
    image = np.array(image)  # Convert to NumPy array (for OpenCV)
    
    # Run YOLOv8 model
    results = yolo_model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls.item())]  # Object label
            confidence = float(box.conf.item())  # Confidence score
            detections.append(f"{label} (confidence: {confidence:.2f})")

            # Draw bounding boxes
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB before displaying
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="📌 Detected Objects", use_container_width=True)

    # AI Description with LLaMA
    if detections:
        input_text = f"Describe these objects in detail in English: {', '.join(detections)}."
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {"role": "system", "content": "You are an AI assistant providing detailed object descriptions in English."},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.7,   # Increase creativity
            "max_tokens": 512,    # Allow longer responses
            "top_p": 0.95,        # More diverse responses
            "frequency_penalty": 0.3,  # Reduce repetition
            "presence_penalty": 0.4    # Encourage richer descriptions
        }

        try:
            response = requests.post(LM_STUDIO_API_URL, json=payload)
            response_data = response.json()
            description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No description available.")
        except Exception as e:
            description = f"Error fetching response: {str(e)}"

        st.subheader("📜 AI Description:")
        st.write(description)

        # Convert Text to Speech (TTS) using gTTS
        tts = gTTS(description, lang="en")
        tts.save("voice_note.mp3")

        # Play Audio in Streamlit
        st.subheader("🔊 Audio Description:")
        st.audio("voice_note.mp3", format="audio/mp3")

        # Process user query when the button is clicked
        # if query_submitted:
        #     if user_query.lower() in description.lower():
        #         st.success(f"✅ Answer found: {description}")
        #     else:
        #         st.error("❌ No relevant information found.")

else:
    st.warning("⚠️ Please capture or upload an image.")
