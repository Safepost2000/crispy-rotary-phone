import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import easyocr

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

def extract_text_from_image(image_file):
    img = Image.open(image_file)
    reader = easyocr.Reader(['en'])  # Specify the language(s) you want to recognize
    result = reader.readtext(img)
    text = '\n'.join([res[1] for res in result])
    return text

def main():
    st.title("OCR Web App")

    file_type = st.selectbox("Select file type", ("PDF", "Image"))
    uploaded_file = st.file_uploader("Choose a file", type=[".pdf", ".jpg", ".jpeg", ".png"])

    if uploaded_file is not None:
        if file_type == "PDF":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            extracted_text = extract_text_from_image(uploaded_file)

        st.write("Extracted Text:")
        st.text(extracted_text)

if __name__ == "__main__":
    main()
