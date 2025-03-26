import argparse
import json
from pathlib import Path
import yaml
from inference import BankingProcessor
from typing import Dict, List
import email
from email import policy
import io
from typing import Dict, List
import PyPDF2
from docx import Document
#from utils.file_handlers import read_eml_file

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
        try:
            with io.BytesIO(content) as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            print(f"PDF read error: {str(e)}")
        return "\n".join(text)

    def _read_docx(self, content: bytes) -> str:
        """Extract text from DOCX attachments"""
        try:
            with io.BytesIO(content) as f:
                doc = Document(f)
                return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"DOCX read error: {str(e)}")
        return ""

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

def process_single_email(file_path: Path, processor: BankingProcessor, output_dir: Path) -> None:
    """Process single email file and save results"""
    try:
        combined_text = read_eml_file(str(file_path))
        if not combined_text.strip():
            print(f"Skipping empty file: {file_path.name}")
            return

        result = processor.process_email(combined_text)
        output_path = output_dir / f"{file_path.stem}_result.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Processed {file_path.name} → {output_path}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

def main():

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    # Initialize components
    processor = BankingProcessor()

    with open(f"{PROJECT_ROOT}/src/config/paths.yaml") as f:
        paths = yaml.safe_load(f)

    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="Process banking emails")
    parser.add_argument('--input_dir', 
                      type=str, 
                      default=f"{PROJECT_ROOT}/{paths['test']['rawmail']}",
                      help="Directory containing .eml files")
    parser.add_argument('--output_dir',
                      type=str, 
                      default=f"{PROJECT_ROOT}/{paths['result']}",
                      help="Directory to save JSON results")
    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Process all .eml files in directory
    print(f"\nProcessing emails from: {input_dir}")
    for eml_file in input_dir.glob("*.eml"):
        process_single_email(eml_file, processor, output_dir)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()