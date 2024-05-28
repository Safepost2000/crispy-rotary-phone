import streamlit as st
import cv2
from PIL import Image
import pytesseract
import spacy

# Load the image from the user input
image = st.file_uploader("Upload your image", type=["jpg", "png"])

# Preprocess the image
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Extract text using Tesseract OCR
text = pytesseract.image_to_string(thresh)

# Preprocess the extracted text using spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Extract invoice information
invoice_info = []
for ent in doc.ents:
    if ent.label_ == "ORGANIZATION":
        invoice_info.append({"Organization": ent.text})

# Save the extracted information to a .csv file
import csv
with open("invoice_info.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Organization"])
    writer.writerows(invoice_info)

st.success("Invoice information extracted and saved to invoice_info.csv")
