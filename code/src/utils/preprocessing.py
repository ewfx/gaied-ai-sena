import re

def clean_email_text(text):
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove email headers
    return re.split(r'Content-Type: text/plain', text)[-1]