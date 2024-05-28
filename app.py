import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import pandas as pd
import os
import tempfile

def extract_data(image_path):
    # Ensure the image path is valid
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        return None
    
    # Temporarily save the uploaded file to access its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    # Now use the temporary file path for image processing
    preprocessed_img = preprocess_image(temp_file_path)
    
    # Don't forget to delete the temporary file after use
    os.unlink(temp_file_path)
    
    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(preprocessed_img)
    
    return text
    
def preprocess_image_pil(image_path):
    # Use PIL to open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Convert PIL Image object to OpenCV matrix
    img_cv = np.array(img)
    
    return img_cv
    
# Function to preprocess image
#def preprocess_image(image_path):
    # Load image
    #img = cv2.imread(image_path)
    
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
      #      cv2.THRESH_BINARY,11,2)
    
   # return thresh

# Function to perform OCR and extract data
def extract_data(image_path):
    # Preprocess image
    preprocessed_img = preprocess_image(image_path)
    
    # Perform OCR
    text = pytesseract.image_to_string(preprocessed_img)
    
    return text

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
