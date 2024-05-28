import os 
import csv
import cv2
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # Convert the image to grayscale

# Load the OCR model (Chinese+English)
ocr = PaddleOCR(use_gpu=False, lang="en")

def extract_invoice_info(image):
    # Run OCR
    result = ocr.ocr(image)

    # Extract relevant information
    invoice_info = {
        "vendor": "",
        "date": "",
        "total": ""
    }

    for line in result:
        line_text = " ".join([word_info[-1] for word_info in line])
        if "Invoice" in line_text.upper():
            for word_info in line:
                if "Vendor" in word_info[-1].upper():
                    invoice_info["vendor"] = word_info[-1]
                if "Date" in word_info[-1].upper():
                    invoice_info["date"] = word_info[-1]
        if "Total" in line_text.upper():
            for word_info in line:
                if "Total" in word_info[-1].upper():
                    invoice_info["total"] = word_info[-1]

    return invoice_info

def main():
    st.title("Invoice OCR Web App")
    st.write("Upload an invoice image to extract information")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Read the uploaded image
        image = uploaded_file.read()

        # Extract invoice information
        invoice_info = extract_invoice_info(image)

        # Display the extracted information
        st.write("Extracted Invoice Information:")
        st.write("Vendor:", invoice_info["vendor"])
        st.write("Date:", invoice_info["date"])
        st.write("Total:", invoice_info["total"])

        # Save the extracted information to a CSV file
        with open("invoice_info.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Vendor", "Date", "Total"])
            writer.writerow([invoice_info["vendor"], invoice_info["date"], invoice_info["total"]])

        st.write("Invoice information saved to `invoice_info.csv`")

if __name__ == "__main__":
    main()
