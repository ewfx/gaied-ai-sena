from pathlib import Path
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import yaml
from spacy.pipeline import EntityRuler

class BankingProcessor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        with open(f"{self.project_root}/src/config/paths.yaml") as f:
            self.paths = yaml.safe_load(f)
            
        with open(f"{self.project_root}/src/config/settings.yaml") as f:
            self.settings = yaml.safe_load(f)
            
        # Load classifier components
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
            f"{self.project_root}{self.paths['models']['classifier']}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{self.project_root}{self.paths['models']['classifier']}"
        )
        
        # Load NER model with entity rules
        self.ner_model = spacy.load(f"{self.project_root}{self.paths['models']['ner']}")
        self._add_entity_rules()
        
        print("Models loaded successfully")

    def _add_entity_rules(self):
        """Add regex patterns for critical financial entities"""
        patterns = [
            {"label": "LOAN_AMOUNT", "pattern": [{"TEXT": {"REGEX": "^(USD|\$)\s?[\d,]+$"}}]},
            {"label": "ACCOUNT_NUMBER", "pattern": [{"TEXT": {"REGEX": "^\d{10}$"}}]},
            {"label": "LOAN_ID", "pattern": [{"TEXT": {"REGEX": "^LOAN-\d{4}$"}}]}
        ]
        
        if "entity_ruler" not in self.ner_model.pipe_names:
            ruler = self.ner_model.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(patterns)
            print("Added entity rules")

    def _classify_text(self, text):
        """Enhanced classification with error handling"""
        inputs = self.tokenizer(
            text[:512],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        
        return {
            "label": self.classifier_model.config.id2label[pred_idx],
            "confidence": probs[0][pred_idx].item()
        }

    def process_email(self, text):
        try:
            # Classification
            classification = self._classify_text(text)
            print(f"Classified as: {classification['label']} ({classification['confidence']:.2f})")
            
            # Entity Extraction
            doc = self.ner_model(text)
            print(f"Raw entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
            
            # Filter entities by request type
            allowed_entities = self.settings['ner']['entity_map'].get(
                classification['label'], []
            )
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
                if ent.label_ in allowed_entities
            ]
            
            return {
                "type": classification['label'],
                "confidence": classification['confidence'],
                "entities": entities
            }
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {
                "type": "error",
                "confidence": 0.0,
                "entities": []
            }