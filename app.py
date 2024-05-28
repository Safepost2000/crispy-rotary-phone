import streamlit as st
import cv2
import pytesseract
import pandas as pd
import re
import os

# Set up Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Create a Streamlit app
st.title("Invoice OCR Web App")
st.write("Upload an invoice image or PDF file to extract information")

# Create a file uploader
uploaded_file = st.file_uploader("Select an invoice file", type=["jpg", "jpeg", "png", "pdf"])

# Create a button to trigger the OCR process
if st.button("Extract Invoice Information"):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Convert the uploaded file to an image using pdf2image
        from pdf2image import convert_from_path
        images = convert_from_path("temp.pdf")

        # Extract text from the image using Tesseract OCR
        text = []
        for img in images:
            img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(img, config=custom_config)
            text.append(extracted_text)

        # Extract invoice information using regular expressions
        invoice_info = {}
        for page in text:
            # Extract date
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, page)
            if dates:
                invoice_info['Date'] = dates[0]

            # Extract invoice number
            invoice_pattern = r'\bINVOICE\s*NO.\s*(\d+)\b'
            invoice_match = re.search(invoice_pattern, page, re.IGNORECASE)
            if invoice_match:
                invoice_info['Invoice Number'] = invoice_match.group(1)

            # Extract bill to and ship to information
            bill_to_pattern = r'\bBILL\s*TO\s*(.*)\s*SHIP\s*TO\s*(.*)'
            bill_to_match = re.search(bill_to_pattern, page, re.IGNORECASE)
            if bill_to_match:
                invoice_info['Bill To'] = bill_to_match.group(1)
                invoice_info['Ship To'] = bill_to_match.group(2)

            # Extract items and quantities
            item_pattern = r'\b(QTY|Quantity)\s*(\d+)\s*(\w+)\s*(\d+(\.\d+)?)'
            items = re.findall(item_pattern, page)
            if items:
                invoice_info['Items'] = []
                for item in items:
                    invoice_info['Items'].append({
                        'Quantity': item[1],
                        'Item': item[2],
                        'Price': item[3]
                    })

        # Create a Pandas dataframe from the extracted information
        df = pd.DataFrame([invoice_info])

        # Save the dataframe to a CSV file
        df.to_csv("invoice_info.csv", index=False)

        # Display a success message
        st.write("Invoice information extracted successfully! Download the CSV file below.")
        st.markdown("### Download CSV File")
        st.download_button("Download CSV", "invoice_info.csv", "invoice_info.csv", "text/csv")
    else:
        st.write("Please upload an invoice file to extract information")
