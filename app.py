import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import easyocr

st.title("Invoice OCR Web App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])

# Function to extract invoice information
def extract_invoice_info(text):
    lines = text.split('\n')
    invoice_info = {
        "Invoice Number": None,
        "Date": None,
        "Total Amount": None
    }
    for line in lines:
        if "Invoice" in line:
            invoice_info["Invoice Number"] = line.split()[-1]
        if "Date" in line:
            invoice_info["Date"] = line.split()[-1]
        if "Total" in line or "Amount" in line:
            invoice_info["Total Amount"] = line.split()[-1]
    return invoice_info

# Main processing
if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Extracting text...")
        
        # Use pytesseract for OCR
        try:
            text = pytesseract.image_to_string(image)
        except Exception as e:
            st.error(f"Error using pytesseract: {e}")
            text = ""

        # Fallback to EasyOCR if pytesseract fails or for better accuracy on handwritten text
        if not text.strip():
            st.write("Using EasyOCR for better handwritten text recognition...")
            reader = easyocr.Reader(['en'])
            results = reader.readtext(np.array(image))
            text = ' '.join([res[1] for res in results])
        
        # Display extracted text
        st.text_area("Extracted Text", text, height=300)
        
        # Extract and display invoice information
        invoice_info = extract_invoice_info(text)
        df = pd.DataFrame([invoice_info])
        st.write("Extracted Invoice Information:")
        st.dataframe(df)
        
        # Download CSV button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='invoice_info.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an image file.")
