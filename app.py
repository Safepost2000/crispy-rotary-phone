import streamlit as st
import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image

# Configure pytesseract to use the installed tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update this path based on your installation

def preprocess_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def process_image(image_file):
    # Read the image file
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    
    # Check if the image is empty
    if img is None:
        raise ValueError("The uploaded file is not a valid image.")
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    # Perform OCR
    text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
    return text

def main():
    st.title("Advanced Invoice OCR App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an invoice image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)

        # Process the image
        try:
            extracted_text = process_image(uploaded_file)
            # Save text to CSV
            df = pd.DataFrame([extracted_text.split('\n')])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='extracted_text.csv',
                mime='text/csv',
            )
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # Instructions
    st.sidebar.title("Instructions")
    st.sidebar.write("1. Upload an image of an invoice.")
    st.sidebar.write("2. Wait for the OCR to process the image.")
    st.sidebar.write("3. Download the extracted text as a CSV file.")

if __name__ == "__main__":
    main()
