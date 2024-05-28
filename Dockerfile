FROM streamlit/streamlit:latest

# Install Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the rest of your app files
COPY . /app

# Set working directory
WORKDIR /app

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
