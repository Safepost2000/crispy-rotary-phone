import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np

st.title('Invoice OCR Web App')
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Path to the Tesseract executable
uploaded_file = st.file_uploader("Upload Invoice Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    extracted_text = pytesseract.image_to_string(image)
    st.write('Extracted Text:')
    st.write(extracted_text)

    # Save extracted text to a .csv file
    if st.button('Save as CSV'):
        df = pd.DataFrame({'Extracted Text': [extracted_text]})
        df.to_csv('extracted_invoice_info.csv', index=False)
        st.success('Data saved to extracted_invoice_info.csv')
