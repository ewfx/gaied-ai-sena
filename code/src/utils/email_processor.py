import os
import email
from email import policy
from email.parser import BytesParser
from typing import Dict, List

def extract_email_data(eml_path: str, attachment_dir: str = "attachments") -> Dict:
    """Extract email body, headers, and attachments from .eml files"""
    os.makedirs(attachment_dir, exist_ok=True)
    
    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    # Extract basic info
    data = {
        "subject": msg["subject"],
        "from": msg["from"],
        "to": msg["to"],
        "body": "",
        "attachments": []
    }

    # Extract body (plain text first)
    body = None
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body = part.get_content()
                break
    else:
        body = msg.get_content()
    
    data["body"] = body.strip() if body else ""

    # Extract attachments
    for part in msg.iter_attachments():
        if part.get_filename():
            filename = part.get_filename()
            attachment_path = os.path.join(attachment_dir, filename)
            with open(attachment_path, "wb") as f:
                f.write(part.get_payload(decode=True))
            data["attachments"].append(attachment_path)

    return data