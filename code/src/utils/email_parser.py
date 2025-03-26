import email
from email import policy
import io
from typing import Dict, List
import PyPDF2
from docx import Document

class EmailParser:
    def parse_email(self, eml_content: bytes) -> Dict:
        """Parse .eml file content into structured data"""
        msg = email.message_from_bytes(eml_content, policy=policy.default)
        
        return {
            "body": self._get_email_body(msg),
            "attachments": self._get_attachments(msg)
        }

    def _get_email_body(self, msg) -> str:
        """Extract main text body from email"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()
        return body.strip()

    def _get_attachments(self, msg) -> List[Dict]:
        """Extract and process attachments"""
        attachments = []
        for part in msg.iter_attachments():
            if part.get_filename():
                content = part.get_payload(decode=True)
                attachments.append({
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "content": content,
                    "text": self._extract_attachment_text(content, part.get_content_type())
                })
        return attachments

    def _extract_attachment_text(self, content: bytes, content_type: str) -> str:
        """Extract text from common attachment types"""
        try:
            if content_type == "application/pdf":
                return self._read_pdf(content)
            elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                 "application/msword"]:
                return self._read_docx(content)
            elif content_type == "text/plain":
                return content.decode("utf-8")
            else:
                return ""
        except Exception as e:
            print(f"Error processing attachment: {str(e)}")
            return ""

    def _read_pdf(self, content: bytes) -> str:
        """Extract text from PDF attachments"""
        text = []
        with PyPDF2.PdfReader(io.BytesIO(content)) as reader:
            for page in reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)

    def _read_docx(self, content: bytes) -> str:
        """Extract text from DOCX attachments"""
        doc = Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])