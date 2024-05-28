import os
import csv
import cv2
import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
import numpy as np

from django.shortcuts import get_object_or_404

def product_detail(request, id):
    product = get_object_or_404(Product, id=id)
    return render(request, 'product_detail.html', {'product': product})

def extract_invoice_info(image):
    # Run OCR
    ocr = PaddleOCR(use_gpu=False, lang="en")
    result = ocr.ocr(image)

    # Extract relevant information
    invoice_info = {
        "vendor": "",
        "date": "",
        "total": ""
    }

    for line in result:
        line_text = " ".join(word_info[-1] for word_info in line if isinstance(word_info, tuple))
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
        image_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

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
