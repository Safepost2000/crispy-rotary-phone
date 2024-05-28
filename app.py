import streamlit as st
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(page_title="Invoice OCR", layout="wide")

# Function to perform OCR on the uploaded image
def extract_invoice_data(image):
    # Convert the image to grayscale
    gray = np.array(image.convert('L'))

    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(gray)

    # Split the text into lines
    lines = text.split('\n')

    # Extract the relevant information from the lines
    invoice_data = {
        'Invoice Number': '',
        'Total Amount': '',
        'Date': '',
        'Vendor Name': ''
    }

    for line in lines:
        if 'Invoice Number' in line:
            invoice_data['Invoice Number'] = line.split(':')[1].strip()
        elif 'Total' in line or 'Amount' in line:
            invoice_data['Total Amount'] = line.split(':')[1].strip()
        elif 'Date' in line:
            invoice_data['Date'] = line.split(':')[1].strip()
        elif 'Vendor' in line or 'Company' in line:
            invoice_data['Vendor Name'] = line.split(':')[1].strip()

    return invoice_data

# Streamlit app
st.title("Invoice OCR")

# File upload
uploaded_file = st.file_uploader("Choose an invoice image", type=['jpg', 'png', 'pdf'])

if uploaded_file is not None:
    # Load the image
    if uploaded_file.type == 'application/pdf':
        pages = convert_from_bytes(uploaded_file.read())
        image = pages[0]
    else:
        image = Image.open(uploaded_file)

    # Extract invoice data
    invoice_data = extract_invoice_data(image)

    # Display the extracted information
    st.subheader("Invoice Details")
    for key, value in invoice_data.items():
        st.write(f"{key}: {value}")

    # Display the uploaded image
    st.subheader("Uploaded Invoice")
    st.image(image, use_column_width=True)
    
