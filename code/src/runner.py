from pathlib import Path
from pipelines.data_generation import DataGenerator
from pipelines.training import ModelTrainer
from pipelines.annotation import Annotator


## Generate Training Data

# file_num = 500
# dg = DataGenerator()
# dg.generate_dataset(file_num)

## Annotate Data

# an = Annotator()
# an.process_labels()

## Training Classifier and ner Models
# trainer = ModelTrainer()
# trainer.train_classifier()
# trainer.train_ner()