import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import pandas as pd
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    return thresh

# Function to perform OCR and extract data
def extract_data(image_path):
    # Preprocess image
    preprocessed_img = preprocess_image(image_path)
    
    # Perform OCR with PyTesseract
    text_tess = pytesseract.image_to_string(preprocessed_img)
    
    # Perform OCR with EasyOCR for better handwriting recognition
    results_easy = reader.readtext(image_path)
    text_easy = "\n".join([item[1] for item in results_easy])
    
    # Combine texts from both OCR engines
    combined_text = f"{text_tess}\n{text_easy}"
    
    return combined_text

# Streamlit UI
st.title('Invoice Data Extraction')

uploaded_file = st.file_uploader("Choose an invoice image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Invoice', use_column_width=True)
    
    # Extract text from the image
    extracted_text = extract_data(uploaded_file.name)
    st.write(extracted_text)
    
    # Process and save extracted data to a.csv file
    # Assuming the extracted text contains invoice details in a structured format
    # You might need to implement custom logic to parse the text and save it to a.csv file
    pass
