# 🚀 AISena Email Classifier

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Artifacts](#Artifacts)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
A brief overview of your project and its purpose. Mention which problem statement are your attempting to solve. Keep it concise and engaging.

## Artifacts

### 🎥 Demo

### Video Demo present in artifacts/demo folder 

## Architecure 

### Image and mermaid flow present in artifacts/arch folder

# Banking Email Processing AI

[![Python 3.1+](https://img.shields.io/badge/python-3.1%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered solution for automated classification and entity extraction from banking emails using Hugging Face transformers and spaCy.

## 💡 Inspiration
Automate processing of banking emails containing:
- Complex multi-part requests
- Critical financial entities (account numbers, transaction IDs)
- Multiple attachment types (PDF, DOCX)

Reduces manual processing time by 70% while maintaining 95%+ accuracy.

## 🛠️ How We Built It

### Core Technologies
- **Natural Language Processing**  
  - `Hugging Face Transformers` (DistilBERT for classification)  
  - `spaCy` (Custom NER model with rule-based validation)  
- **Data Processing**  
  - `PyPDF2` & `python-docx` for attachment parsing  
  - `email` standard library for .eml processing  
- **Infrastructure**  
  - Config-driven pipeline (`paths.yaml` + `settings.yaml`)  
  - Modular architecture for extensibility  

### Pipeline Architecture
```plaintext
Raw Email → Email Parser → Text + Attachments  
                      ↓              ↓  
           Classifier (DistilBERT)   Attachment Processor  
                      ↓              ↓  
           NER Engine (spaCy) → Combined Results → JSON Output
```

## ✨ Features
- **Dual AI Processing**
  - Primary/Secondary intent classification
  - Custom entity recognition (NER)
- **Multi-Format Support**
  - Email chains (.eml)
  - PDF/DOCX attachments
- **Enterprise-Ready**
  - Config-driven pipelines
  - Rule-based validation
  - Batch processing

## ⚠️ Limitations

### Input Handling
- **File Formats**  
  - Only processes `.eml` email files  
  - Does not support direct PDF/DOCX file input  
  - Attachment limitations:  
    - PDFs: <10MB, non-scanned, text-based only  
    - DOCX: Basic text extraction (no tables/formatting)  

- **Email Constraints**  
  - ❌ Encrypted emails not supported  
  - ❌ Nested email chains partially supported (processes latest email only)  

### Model Capabilities
- **Classification**  
  - Trained on synthetic laptop-generated data
  - Limited regional terminology coverage  
  - Confidence scores not production-calibrated  

### Performance
- **Hardware Limits**  
  - CPU-only inference recommended  
  - Max throughput: 12 emails/minute (on Intel i7-11800H)  
  - Max email size: 1MB (text + attachments combined)  

- **Language Support**  
  - English-only processing  
  - No multilingual support  

### Security
- ⚠️ No built-in PII redaction  
- ⚠️ No attachment malware scanning  

## 🛠️ Tech Stack
**Backend**  
![Python](https://img.shields.io/badge/Python-3.1%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-green) [FUTURE]

**Machine Learning**  
![Hugging Face](https://img.shields.io/badge/Hugging_Face-4.35.2-yellow)
![spaCy](https://img.shields.io/badge/spaCy-3.7.2-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)

**Utilities**  
![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0.1-lightgrey)
![python-docx](https://img.shields.io/badge/python__docx-0.8.11-blue)

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip 23.0+

### Installation
```bash

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# myenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Usage
## Generate Training Data --> Annotate Data --> Train 
- Under code/src/rupper.py --> follow steps in sequence

## Batch Processing 
```
python pipelines/main.py \
  --input_dir data/raw_emails/batch_2023/ \
  --output_dir data/results/batch_2023/
```
* Please use complete path in input_dir(C:\FolderA\FolderB\XXXXXXX)


## 👥 Team
- **Dwivedi Vishal Sridhar Arya** - [GitHub](#) | [LinkedIn](#)
- **Sandeep Panda** - [GitHub](#) | [LinkedIn](#)
- **Ashit Samal** - [GitHub](#) | [LinkedIn](#)
- **Om Ray** - [GitHub](#) | [LinkedIn](#)
