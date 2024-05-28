import streamlit as st
import easyocr
import csv

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# File uploader
uploaded_file = st.file_uploader("Choose an invoice image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the image
    image = uploaded_file.read()

    # Perform OCR
    result = reader.readtext(image)

    # Extract text and display
    extracted_text = ' '.join([text[-2] for text in result])
    st.write("Extracted Text:")
    st.write(extracted_text)

    # Write to CSV
    if st.button("Save to CSV"):
        with open("invoice_data.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Text"])
            writer.writerow([extracted_text])

        st.success("CSV file generated successfully!")
        st.download_button(
            label="Download CSV",
            data=open("invoice_data.csv", "rb").read(),
            file_name="invoice_data.csv",
            mime="text/csv",
        )
