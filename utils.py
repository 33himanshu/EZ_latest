from pdfminer.high_level import extract_text

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        # Save uploaded file to disk temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        return extract_text("temp_uploaded.pdf")

    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")

    else:
        return "Unsupported file type."
