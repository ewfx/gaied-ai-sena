from pathlib import Path
from typing import Optional
from email_parser import EmailParser

def read_eml_file(file_path: str) -> str:
    """Read .eml file and return combined text (body + attachments)"""
    try:
        with open(file_path, "rb") as f:
            eml_content = f.read()
        parsed_email = EmailParser().parse_email(eml_content)
        combined_text = parsed_email["body"]
        
        # Add attachment texts
        for att in parsed_email["attachments"]:
            combined_text += f"\n[ATTACHMENT: {att['filename']}]\n{att['text']}"
        
        return combined_text
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return ""