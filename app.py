import streamlit as st
import tempfile
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Function to preprocess image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    return thresh

# Function to perform OCR and extract data
def extract_data(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    # Use the temporary file path for image processing
    preprocessed_img = preprocess_image(temp_file_path)
    
    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_img)
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    return text

# Streamlit UI
st.title('Invoice Data Extraction')

uploaded_file = st.file_uploader("Choose an invoice image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Invoice', use_column_width=True)
    
    # Extract text from the image
    extracted_text = extract_data(uploaded_file)
    st.write(extracted_text)
