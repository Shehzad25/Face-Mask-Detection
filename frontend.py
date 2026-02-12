import streamlit as st
import requests

API_URL = "http://51.20.87.138:8000/predict"

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("😷 Face Mask Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            data = response.json()

            if data["faces_detected"] == 0:
                st.warning("No face detected 😕")
            else:
                label = data["predictions"][0]["label"]

                if label == "Mask":
                    st.success("😷 Mask")
                else:
                    st.error("❌ Without Mask")
        else:
            st.error("Prediction failed. Please try again.")
