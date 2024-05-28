import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update with your Tesseract path

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Apply some preprocessing techniques
    gray = cv2.medianBlur(gray, 3)
    return gray

def extract_text(image):
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

def extract_handwritten_text(image):
    # For handwritten text, you may need to adjust OCR configuration
    custom_config = r'--oem 1 --psm 3'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def process_invoice(image):
    preprocessed_image = preprocess_image(image)
    text = extract_text(preprocessed_image)
    handwritten_text = extract_handwritten_text(preprocessed_image)
    return text, handwritten_text

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['Type', 'Content'])
    df.to_csv(filename, index=False)

def main():
    st.title('Invoice OCR Web App')
    st.write("Upload an invoice image to extract text and handwritten notes")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Invoice', use_column_width=True)
        st.write("Extracting text...")
        text, handwritten_text = process_invoice(image)

        st.subheader('Extracted Text')
        st.write(text)

        st.subheader('Extracted Handwritten Text')
        st.write(handwritten_text)

        if st.button('Save to CSV'):
            data = [['Text', text], ['Handwritten Text', handwritten_text]]
            save_to_csv(data, 'invoice_data.csv')
            st.success('Data saved to invoice_data.csv')

if __name__ == "__main__":
    main()
