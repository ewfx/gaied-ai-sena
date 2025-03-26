from transformers import pipeline
import spacy

def validate_models():
    # Test classifier
    classifier = pipeline("text-classification", model="code/src/models/classifier")
    print("Classifier test:", classifier("Approved loan of USD 1M to ABTB LLC"))
    
    # Test NER
    nlp = spacy.load("code/src/models/ner")
    doc = nlp("Borrower: Global Investments Inc. Account #: 1234567890")
    print("NER test:", [(ent.text, ent.label_) for ent in doc.ents])

if __name__ == "__main__":
    validate_models()