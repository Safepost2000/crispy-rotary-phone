import streamlit as st
import easyocr
from PIL import Image, ImageEnhance, ImageOps
import fitz  # PyMuPDF
import numpy as np
import tempfile
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords

# Ensure necessary NLTK data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image):
    image = ImageOps.grayscale(image)  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Enhance contrast
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)  # Resize to improve accuracy
    return image

def extract_text_from_image(image):
    image_np = np.array(preprocess_image(image))  # Convert PIL image to NumPy array after preprocessing
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

def extract_information(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]  # Remove punctuation
    words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords

    tagged_words = pos_tag(words)
    entities = ne_chunk(tagged_words)

    info = {
        'Names': [],
        'Organizations': [],
        'Locations': [],
        'Dates': [],
        'Other': []
    }

    for entity in entities:
        if isinstance(entity, nltk.Tree):
            entity_label = entity.label()
            entity_text = ' '.join([child[0] for child in entity])
            if entity_label == 'PERSON':
                info['Names'].append(entity_text)
            elif entity_label == 'ORGANIZATION':
                info['Organizations'].append(entity_text)
            elif entity_label == 'GPE':
                info['Locations'].append(entity_text)
            elif entity_label == 'DATE':
                info['Dates'].append(entity_text)
            else:
                info['Other'].append(entity_text)
    
    return info

def display_info(info):
    st.subheader("Extracted Information")
    for key, values in info.items():
        st.write(f"**{key}:**")
        for value in values:
            st.write(f"- {value}")

def main():
    st.title("OCR Web App with EasyOCR and NLTK")

    st.sidebar.title("Upload your files")
    file = st.sidebar.file_uploader("Choose an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if file is not None:
        file_type = file.type
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.write("Extracting text...")
            try:
                text = extract_text_from_image(image)
                st.write(text)
                info = extract_information(text)
                display_info(info)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        elif file_type == "application/pdf":
            st.write("Extracting text from PDF...")
            try:
                text = extract_text_from_pdf(file)
                st.write(text)
                info = extract_information(text)
                display_info(info)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()
