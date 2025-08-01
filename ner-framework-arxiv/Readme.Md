
# NER Framework for Scientific Literature Analysis

## Overview

This repository contains the implementation of a Named Entity Recognition (NER) framework specifically designed for extracting hardware specifications, cloud platforms, and software tools from scientific literature (ArXiv papers). The framework combines BERT-based classification, SciBERT NER models, and Large Language Models (LLMs) to achieve high-accuracy entity extraction.

## Key Features

- **PDF Processing**: Automated extraction and preprocessing of text from scientific PDFs  
- **Multi-stage Pipeline**:
  - Binary classification using fine-tuned BERT
  - Embedding similarity matching for relevant sentences
  - SciBERT-based NER for entity recognition
  - LLM-based extraction for low-confidence predictions
- **Affiliation Extraction**: Specialized module for extracting and normalizing author affiliations  
- **Entity Categories**:
  - Hardware devices (GPUs, CPUs)
  - Device memory specifications
  - Device count information
  - Cloud platforms
  - Software libraries and frameworks

## Architecture

```

Input PDF → Text Extraction → Preprocessing → Binary Classification →
Embedding Similarity → NER Model → LLM Enhancement → JSON Output

````

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab or local environment

### Setup

1. Clone the repository:

```bash
git clone https://github.com/hubstack8/InfraKG.git
cd InfraKG
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download pre-trained models:

```bash
# Download links will be provided separately
# Place models in the appropriate directories
```

## Usage

### Quick Start

```python
from src.pipeline.main_pipeline import NERPipeline

# Initialize pipeline
pipeline = NERPipeline(
    bert_model_path="path/to/bert/model",
    ner_model_path="path/to/ner/model",
    api_key="your-api-key"
)

# Process a single PDF
results = pipeline.process_pdf("path/to/paper.pdf", "path/to/metadata.json")
```

### Batch Processing

```python
from scripts.run_pipeline import batch_process

batch_process(
    pdf_folder="path/to/pdfs",
    output_folder="path/to/outputs"
)
```

## Model Details

### Binary Classifier (BERT)

* Base model: `bert-base-uncased`
* Fine-tuned on scientific text for relevance classification
* Identifies sentences containing hardware/software mentions

### NER Model (SciBERT)

* Base model: `allenai/scibert_scivocab_uncased`
* Fine-tuned for scientific entity recognition
* Supports BIO tagging scheme

### LLM Enhancement

* Model: DeepSeek Chat v3
* Used for extracting entities from low-confidence predictions
* Temperature: 0.1 for consistent outputs


