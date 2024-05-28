import streamlit as st
import easyocr
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import tempfile

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text_from_image(image):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    result = reader.readtext(image_np)
    return ' '.join([text for _, text, _ in result])

def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file.seek(0)
        doc = fitz.open(tmp_file.name)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += extract_text_from_image(img)
    return text

def main():
    st.title("OCR Web App with EasyOCR")

    st.sidebar.title("Upload your files")
    file = st.sidebar.file_uploader("Choose an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if file is not None:
        file_type = file.type
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Extracting text...")
            text = extract_text_from_image(image)
            st.write(text)
        elif file_type == "application/pdf":
            st.write("Extracting text from PDF...")
            text = extract_text_from_pdf(file)
            st.write(text)

if __name__ == "__main__":
    main()
