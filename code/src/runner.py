from pathlib import Path
from pipelines.data_generation import DataGenerator
from pipelines.training import ModelTrainer
from pipelines.annotation import Annotator

path = "code/src/data/raw_emails"
# dg = DataGenerator()
# dg.generate_dataset(500)

# an = Annotator()
# an.process_labels()


trainer = ModelTrainer()
trainer.train_classifier()
trainer.train_ner()