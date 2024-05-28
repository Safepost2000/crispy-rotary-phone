import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
import easyocr
from pdf2image import convert_from_path
import tempfile
import os

st.title("Invoice OCR Web App")

# File uploader
uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

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

def pdf_to_images(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf.flush()
        images = convert_from_path(temp_pdf.name)
    os.remove(temp_pdf.name)
    return images

# Main processing
if uploaded_file is not None:
    try:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            images = pdf_to_images(uploaded_file.read())
        else:
            images = [Image.open(uploaded_file)]

        all_text = ""
        for image in images:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Extracting text...")

            text = ""
            if pytesseract.get_tesseract_version() is not None:
                try:
                    text = pytesseract.image_to_string(image)
                except Exception as e:
                    st.error(f"Error using pytesseract: {e}")

            if not text.strip():
                st.write("Using EasyOCR for better handwritten text recognition...")
                reader = easyocr.Reader(['en'])
                try:
                    results = reader.readtext(np.array(image))
                    text = ' '.join([res[1] for res in results])
                except Exception as e:
                    st.error(f"An error occurred with EasyOCR: {e}")
                    text = ""

            all_text += text + "\n"

        st.text_area("Extracted Text", all_text, height=300)

        invoice_info = extract_invoice_info(all_text)
        df = pd.DataFrame([invoice_info])
        st.write("Extracted Invoice Information:")
        st.dataframe(df)

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
    st.write("Please upload an image or PDF file.")
