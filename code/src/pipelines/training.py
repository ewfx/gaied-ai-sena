import random
import re
import spacy
import yaml
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Features, Value, ClassLabel
import json
from pathlib import Path
import warnings
from transformers import logging
from spacy.scorer import Scorer
from spacy.util import minibatch
from spacy.training import Example
from tqdm import tqdm

# Suppress warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ModelTrainer:
    project_root = Path(__file__).parent.parent.parent
    MODEL_NAME = "en_core_web_lg"
    VALIDATION_SPLIT = 0.2
    EPOCHS = 10
    BATCH_SIZE = 8
    def __init__(self):
        with open(f"{self.project_root}/src/config/paths.yaml") as f:
            self.paths = yaml.safe_load(f)
            
        with open(f"{self.project_root}/src/config/settings.yaml") as f:
            self.settings = yaml.safe_load(f)

    # ------------------------
    # TRAIN CLASSIFIER MODEL
    # ------------------------
    def validate_json_entry(self,entry: dict) -> bool:
        required_fields = {
            "file": str,
            "body": str,
            "request_type": str,
            "entities": str
        }
        
        for field, dtype in required_fields.items():
            if field not in entry:
                print(f"Missing field: {field}")
                return False
            if not isinstance(entry[field], dtype):
                print(f"Invalid type for {field}: {type(entry[field])} (expected {dtype})")
                return False
        
        try:
            entities = json.loads(entry["entities"])
            if not isinstance(entities, list):
                raise ValueError("Entities is not a list")
            for entity in entities:
                if not isinstance(entity, dict) or "entity" not in entity or "value" not in entity:
                    raise ValueError("Invalid entity structure")
        except Exception as e:
            print(f"Invalid entities in {entry['file']}: {str(e)}")
            return False
        
        return True

    def load_and_process_dataset(self,json_files):
        dataset = load_dataset(
            'json',
            data_files=json_files,
            features=Features({
                "file": Value("string"),
                "body": Value("string"),
                "request_type": Value("string"),
                "entities": Value("string")
            })
        )
        
        def parse_entities(example):
            try:
                entities = json.loads(example["entities"])
                if all(isinstance(e, dict) and 'entity' in e and 'value' in e for e in entities):
                    return {"entities": entities}
                return {"entities": []}
            except json.JSONDecodeError:
                return {"entities": []}

        return dataset.map(parse_entities)

    # def prepare_labels(self,dataset):
    #     unique_labels = sorted(set(dataset["train"]["request_type"]))
    #     class_label = ClassLabel(names=unique_labels)
    #     return dataset.cast_column("request_type", class_label), class_label

    def prepare_labels(self, dataset):
        # Define fixed label mapping
        label_map = {
            "loan_approval": 0,
            "share_adjustment": 1,
            "other": 2
        }
        features = Features({
            "file": Value("string"),
            "body": Value("string"),
            "request_type": ClassLabel(names=list(label_map.keys())),
            "entities": Value("string")
        })
        return dataset.cast(features), label_map

    
    def train_classifier(self):
        try:
            # Define FIXED label mapping
            LABEL_MAP = {
                "loan_approval": 0,
                "share_adjustment": 1,
                "other": 2
            }

            # Load and filter data
            label_path = Path(self.paths["data"]["labels"])
            valid_files = []
            for f in label_path.glob("*.json"):
                data = json.load(open(f))
                if data["request_type"] in LABEL_MAP:
                    valid_files.append(str(f))
                else:
                    print(f"Removing invalid label: {data['request_type']} in {f.name}")

            if not valid_files:
                raise ValueError("No valid training files after filtering")

            # Prepare dataset with fixed labels
            features = Features({
                "file": Value("string"),
                "body": Value("string"),
                "request_type": ClassLabel(names=list(LABEL_MAP.keys())),
                "entities": Value("string")
            })
            dataset = load_dataset('json', data_files=valid_files, features=features)

            # Encode labels to integers
            def encode_labels(example):
                return {"labels": example["request_type"]}
            dataset = dataset.map(encode_labels)

            # Tokenize text and retain labels
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            def tokenize_fn(examples):
                tokenized = tokenizer(
                    examples["body"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
                tokenized["labels"] = examples["labels"]
                return tokenized
            dataset = dataset.map(tokenize_fn, batched=True)

            # Split dataset
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

            # Initialize model
            model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(LABEL_MAP),
                id2label={v: k for k, v in LABEL_MAP.items()},
                label2id=LABEL_MAP
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.paths['models']['classifier'],
                num_train_epochs=5,
                per_device_train_batch_size=16,
                evaluation_strategy='epoch',
                logging_dir='./logs',
                seed=42,
                learning_rate=2e-5,
                remove_unused_columns=True
            )

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
            )

            # Train and save
            trainer.train()
            trainer.save_model(self.paths['models']['classifier'])
            tokenizer.save_pretrained(self.paths['models']['classifier'])

        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            exit(1)

    # ------------------------
    # TRAIN NER MODEL
    # ------------------------
    
    def create_ner_model(self):
        """Create fresh NER model with embeddings from en_core_web_lg"""
        # Create blank English model
        nlp = spacy.blank("en")
        print("Created blank English model")

        # Add vectors from en_core_web_lg
        try:
            vec_model = spacy.load("en_core_web_lg")
            nlp.vocab.vectors = vec_model.vocab.vectors
            print("Loaded word vectors from en_core_web_lg")
        except Exception as e:
            print(f"Couldn't load vectors: {str(e)}")
        
        # Add fresh NER component
        ner = nlp.add_pipe("ner", last=True)
        print("Added new NER component")
        
        return nlp

    def parse_entities(self,entity_str):
        """Safely parse JSON entity string with validation"""
        try:
            entities = json.loads(entity_str)
            if not isinstance(entities, list):
                raise ValueError("Entities should be a list")
                
            valid_entities = []
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                if "entity" not in ent or "value" not in ent:
                    continue
                if not isinstance(ent["value"], str):
                    continue
                valid_entities.append({
                    "entity": str(ent["entity"]),
                    "value": str(ent["value"])
                })
            return valid_entities
        except Exception as e:
            print(f"Entity parsing error: {str(e)}")
            return []

    def validate_entities(self,nlp):
        """Check entity-text alignment before training"""
        errors = []
        for file in Path(self.paths["data"]["labels"]).glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                text = data["body"]
                entities = json.loads(data["entities"])
                
                for ent in entities:
                    matches = self.find_entity_positions(nlp, text, ent["value"])
                    if not matches:
                        errors.append(f"{file.name}: No match for '{ent['value']}'")
        
        if errors:
            print(f"\nCritical Data Issues ({len(errors)}):")
            for err in errors[:10]:
                print(f" - {err}")
            raise ValueError("Fix data mismatches before training")

    # def find_entity_positions(self,nlp, text, value):
    #     """Flexible entity matching with text normalization"""
    #     clean_text = re.sub(r'\s+', ' ', text).strip().lower()
    #     clean_value = re.sub(r'\s+', ' ', value).strip().lower()
        
    #     start = clean_text.find(clean_value)
    #     if start == -1:
    #         return []
        
    #     # Map back to original text positions
    #     original_start = text.lower().find(clean_value, start)
    #     original_end = original_start + len(value)
    #     return [(original_start, original_end)]

    def find_entity_positions(self, nlp, text, value):
        """Robust entity matching with token alignment"""
        doc = nlp.make_doc(text)
        value_doc = nlp.make_doc(value)
        
        # Find matching token sequences
        for i in range(len(doc) - len(value_doc) + 1):
            if doc[i:i+len(value_doc)].text.lower() == value_doc.text.lower():
                start = doc[i].idx
                end = doc[i+len(value_doc)-1].idx + len(doc[i+len(value_doc)-1].text)
                return [(start, end)]
        return []

    def prepare_training_data(self,nlp):
        """Convert serialized entities to spaCy format"""
        train_data = []
        error_log = []
        
        for label_file in Path(self.paths["data"]["labels"]).glob("*.json"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                text = data.get("body", "").strip()
                if not text:
                    error_log.append(f"{label_file.name}: Empty text")
                    continue
                    
                entities = self.parse_entities(data.get("entities", "[]"))
                if not entities:
                    error_log.append(f"{label_file.name}: No valid entities")
                    continue
                    
                spacy_entities = []
                for ent in entities:
                    matches = self.find_entity_positions(nlp, text, ent["value"])
                    if not matches:
                        error_log.append(f"{label_file.name}: No match for '{ent['value']}'")
                        continue
                    
                    start, end = matches[0]
                    spacy_entities.append((start, end, ent["entity"].upper()))
                
                if spacy_entities:
                    train_data.append((text, {"entities": spacy_entities}))
                    
            except Exception as e:
                error_log.append(f"{label_file.name}: {str(e)}")
        
        # Print error summary
        print(f"\nData preparation errors ({len(error_log)}):")
        for err in error_log[:5]:
            print(f" - {err}")
        if len(error_log) > 5:
            print(f" - ...and {len(error_log)-5} more errors")
        
        if not train_data:
            raise ValueError("No valid training examples found")
        
        # Show sample data
        print("\nFirst training example:")
        print(f"Text: {train_data[0][0]}")
        print(f"Entities: {train_data[0][1]['entities']}")
        
        random.shuffle(train_data)
        split = int(len(train_data) * self.VALIDATION_SPLIT)
        return train_data[split:], train_data[:split]


    def train_ner(self):
        nlp = self.create_ner_model()
        self.validate_entities(nlp)  # Added validation
        
        # Add NER component
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")
        
        # Load and validate data
        try:
            train_data, val_data = self.prepare_training_data(nlp)
            # Training configuration
            nlp.initialize()
            optimizer = nlp.create_optimizer()
            
            # Convert to spaCy examples
            train_examples = [
                Example.from_dict(nlp.make_doc(text), annotations)
                for text, annotations in train_data
            ]
            
            val_examples = [
            Example.from_dict(nlp.make_doc(text), annotations)
            for text, annotations in val_data
                ]
        
            # Training setup
            optimizer = nlp.initialize()
            best_f1 = 0.0
            print("\nStarting training...")

            # Training loop
            best_f1 = 0.0
            print("\nStarting training...")
            
            for epoch in range(self.EPOCHS):
                random.shuffle(train_examples)
                losses = {}
                
                # Batch training with progress
                batches = minibatch(train_examples, size=self.BATCH_SIZE)
                with tqdm(total=len(train_examples), desc=f"Epoch {epoch+1}") as pbar:
                    for batch in batches:
                        nlp.update(
                            batch,
                            losses=losses,
                            drop=0.2,
                            sgd=optimizer
                        )
                        pbar.update(len(batch))
                
                # Validation
                scores = nlp.evaluate(val_examples)
                print(f"Epoch {epoch+1} | Loss: {losses.get('ner', 0):.2f} | "
                    f"F1: {scores['ents_f']:.2f}")
                
                if scores['ents_f'] > best_f1:
                    nlp.to_disk("best_ner_model")
                    best_f1 = scores['ents_f']
                    print(f"New best F1: {best_f1:.2f}")

            nlp.to_disk(self.paths['models']['ner'])

            print("Training complete!")
        except ValueError as e:
            print(f"\nFatal error: {str(e)}")
            print("Possible fixes:")
            print("1. Check entity values exist in their corresponding texts")
            print("2. Verify JSON entity formatting")
            print("3. Ensure at least one valid training example exists")
            return
    
    def _load_training_data(self):
        # Load processed.spacy
        return []