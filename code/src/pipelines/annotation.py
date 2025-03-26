from spacy.tokens import DocBin
import json
from pathlib import Path
import yaml
import re
from typing import Dict, List
import pdfplumber
from docx import Document
from utils.email_processor import extract_email_data

class Annotator:

    # Regex patterns for entity extraction
    project_root = Path(__file__).parent.parent.parent
    ENTITY_PATTERNS = {
    "borrower": r"(Borrower|Client|Account Holder)[:\s=]+(.+?)\n",
    "amount": r"(Amount|Total):\s([A-Z]{0,3}\s?\$?\$?\s?[\d,]+)",
    "effective_date": r"(Effective Date|Date of Execution)[:\s=]+(\d{2}-[A-Za-z]{3}-\d{4})",
    "loan_id": r"(Loan ID|Reference Number)[:\s=]+(LOAN-\d{4})",
    "account_number": r"(Account #|Account Number)[:\s=]+(\d{10,12})"
        }

    # Request type indicators
    REQUEST_KEYWORDS = {
    "loan_approval": ["approv", "sanction", "authoriz", "loan agreement"],
    "share_adjustment": ["share adjustment", "equity rebalanc", "stock split"],
    "mortgage": ["mortgage", "lien", "property", "collateral"],
    "esop": ["esop", "stock option", "vesting", "equity plan"]
        }
    
    def __init__(self):
        with open("{self.project_root}/src/config/paths.yaml") as f:
            self.paths = yaml.safe_load(f)

    def extract_text_from_pdf(self,pdf_path: str) -> str:
        """Extract text from PDF attachments"""
        try:
            with pdfplumber.open(pdf_path) as pdf:  
                return "\n".join(page.extract_text() for page in pdf.pages)
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {str(e)}")
            return ""

    def extract_text_from_docx(self,docx_path: str) -> str:
        """Extract text from DOCX attachments"""
        try:
            doc = Document(docx_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {docx_path}: {str(e)}")
            return ""

    def extract_entities(self,text: str) -> List[Dict]:
        """Advanced entity extraction using regex patterns"""
        entities = []
        text = text.replace("\r", "").replace("\t", " ")  # Clean text
        
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Prefer the longer match group if multiple captured groups
                value = max(match, key=len).strip()
                entities.append({
                    "entity": entity_type,
                    "value": value
                })
        
        return entities
    
    def auto_label_email(self,eml_path: str) -> Dict:
        """Enhanced auto-labeling with attachment processing"""
        email_data = extract_email_data(eml_path)
        
        # Extract entities from body and attachments
        all_entities = self.extract_entities(email_data["body"])
        
        # Process attachments
        for att_path in email_data["attachments"]:
            if att_path.endswith(".pdf"):
                text = self.extract_text_from_pdf(att_path)
            elif att_path.endswith((".doc", ".docx")):
                text = self.extract_text_from_docx(att_path)
            else:
                continue
                
            all_entities += self.extract_entities(text)
        
        # Deduplicate entities
        seen = set()
        unique_entities = []
        for ent in all_entities:
            key = (ent["entity"], ent["value"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)
        
        return {
            "file": eml_path,
            "body": email_data["body"],
            "request_type": self.classify_request_type(email_data["body"], email_data["attachments"]),
            "entities": json.dumps(unique_entities)
        }

    def classify_request_type(self,body: str, attachments: List[str]) -> str:
        """Enhanced request type classification"""
        combined_text = body.lower()
        
        # Check attachments
        for att_path in attachments:
            if att_path.endswith(".pdf"):
                combined_text += self.extract_text_from_pdf(att_path).lower()
            elif att_path.endswith((".doc", ".docx")):
                combined_text += self.extract_text_from_docx(att_path).lower()
        
        # Determine request type
        scores = {req: 0 for req in self.REQUEST_KEYWORDS}  
        for req_type, keywords in self.REQUEST_KEYWORDS.items():  
            for kw in keywords:  
                scores[req_type] += body.count(kw)  
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"

    def process_labels(self):
        input_dir = Path(self.paths['data']['raw'])
        output_dir = Path(self.paths['data']['labels'])
        
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all emails
        for eml_file in input_dir.glob("*.eml"):
            labels = self.auto_label_email(str(eml_file))
            output_path = output_dir / f"{eml_file.stem}.json"
            
            with open(output_path, "w") as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)
            
            print(f"Processed: {eml_file.name} -> {output_path.name}")

        print(f"\nAnnotation complete! Processed {len(list(input_dir.glob('*.eml')))} files.")
    
    def _validate_entry(self, data):
        # Implement validation checks
        return True
        
    def _convert_to_spacy(self, data):
        db = DocBin()
        # Add spacy conversion logic
        db.to_disk(self.paths['data']['processed']/"train.spacy")