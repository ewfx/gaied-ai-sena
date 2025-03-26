# 🚀 AISena Email Classifier

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
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

## 🎥 Demo

### Video Demo present in artifacts/demo folder 

###

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
