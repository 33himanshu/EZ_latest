from transformers import pipeline

# Load Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    # Limit input size
    limited_text = text[:3000]

    # Generate summary
    summary = summarizer(limited_text, max_length=150, min_length=50, do_sample=False)

    return summary[0]["summary_text"]

# (Optional: keep this test block if you want to run this file independently)
# if __name__ == "__main__":
#     sample_text = """ your test text here """
#     print("Summary:\n", generate_summary(sample_text))
