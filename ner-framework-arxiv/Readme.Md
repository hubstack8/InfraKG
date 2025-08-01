\# NER Framework for Scientific Literature Analysis



\## Overview



This repository contains the implementation of a Named Entity Recognition (NER) framework specifically designed for extracting hardware specifications, cloud platforms, and software tools from scientific literature (ArXiv papers). The framework combines BERT-based classification, SciBERT NER models, and Large Language Models (LLMs) to achieve high-accuracy entity extraction.



\## Key Features



\- \*\*PDF Processing\*\*: Automated extraction and preprocessing of text from scientific PDFs

\- \*\*Multi-stage Pipeline\*\*: 

&nbsp; - Binary classification using fine-tuned BERT

&nbsp; - Embedding similarity matching for relevant sentences

&nbsp; - SciBERT-based NER for entity recognition

&nbsp; - LLM-based extraction for low-confidence predictions

\- \*\*Affiliation Extraction\*\*: Specialized module for extracting and normalizing author affiliations

\- \*\*Entity Categories\*\*:

&nbsp; - Hardware devices (GPUs, CPUs)

&nbsp; - Device memory specifications

&nbsp; - Device count information

&nbsp; - Cloud platforms

&nbsp; - Software libraries and frameworks



\## Architecture



```

Input PDF → Text Extraction → Preprocessing → Binary Classification → 

Embedding Similarity → NER Model → LLM Enhancement → JSON Output

```



\## Installation



\### Prerequisites



\- Python 3.8+

\- CUDA-capable GPU (recommended)

\- Google Colab or local environment



\### Setup



1\. Clone the repository:

```bash

git clone https://github.com/yourusername/ner-framework-arxiv.git

cd ner-framework-arxiv

```



2\. Install dependencies:

```bash

pip install -r requirements.txt

```



3\. Download pre-trained models:

```bash

\# Download links will be provided separately

\# Place models in the appropriate directories

```



\## Usage



\### Quick Start



```python

from src.pipeline.main\_pipeline import NERPipeline



\# Initialize pipeline

pipeline = NERPipeline(

&nbsp;   bert\_model\_path="path/to/bert/model",

&nbsp;   ner\_model\_path="path/to/ner/model",

&nbsp;   api\_key="your-api-key"

)



\# Process a single PDF

results = pipeline.process\_pdf("path/to/paper.pdf", "path/to/metadata.json")

```



\### Batch Processing



```python

from scripts.run\_pipeline import batch\_process



batch\_process(

&nbsp;   pdf\_folder="path/to/pdfs",

&nbsp;   output\_folder="path/to/outputs"

)

```



\## Model Details



\### Binary Classifier (BERT)

\- Base model: `bert-base-uncased`

\- Fine-tuned on scientific text for relevance classification

\- Identifies sentences containing hardware/software mentions



\### NER Model (SciBERT)

\- Base model: `allenai/scibert\_scivocab\_uncased`

\- Fine-tuned for scientific entity recognition

\- Supports BIO tagging scheme



\### LLM Enhancement

\- Model: DeepSeek Chat v3

\- Used for extracting entities from low-confidence predictions

\- Temperature: 0.1 for consistent outputs



\## Output Format



The framework outputs structured JSON with the following schema:



```json

{

&nbsp; "paper\_id": {

&nbsp;   "paper\_title": "Title",

&nbsp;   "paper\_authors": \["Author1", "Author2"],

&nbsp;   "paper\_publish\_year": "2024",

&nbsp;   "paper\_category": \["cs.CL"],

&nbsp;   "scibert\_ner\_results": \[

&nbsp;     {

&nbsp;       "text": "sentence",

&nbsp;       "entity": "NVIDIA V100",

&nbsp;       "label": "Hardware-device",

&nbsp;       "confidence\_score": 0.95

&nbsp;     }

&nbsp;   ],

&nbsp;   "LLMs\_ner\_results": \[

&nbsp;     {

&nbsp;       "sentence": "text",

&nbsp;       "LLMS\_NER": {

&nbsp;         "Hardware-device": \["GPU Name"],

&nbsp;         "Software-Entity": \["TensorFlow", "PyTorch"]

&nbsp;       }

&nbsp;     }

&nbsp;   ],

&nbsp;   "LLMs\_affiliations\_extraction": {

&nbsp;     "Main Organization": "University Name",

&nbsp;     "Sub-Organizations": \[...]

&nbsp;   }

&nbsp; }

}

```



\## Performance



\- Binary Classification Accuracy: ~92%

\- NER F1-Score: ~87%

\- Processing Speed: ~2-3 papers/minute (with GPU)



\## Citation



If you use this framework in your research, please cite:



```bibtex

@article{yourname2024ner,

&nbsp; title={A Multi-Stage NER Framework for Scientific Literature Analysis},

&nbsp; author={Your Name},

&nbsp; journal={Conference/Journal Name},

&nbsp; year={2024}

}

```



\## License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## Contributing



We welcome contributions! Please see our \[Contributing Guidelines](CONTRIBUTING.md) for details.



\## Acknowledgments



\- SciBERT team for the pre-trained scientific language model

\- Hugging Face for the transformers library

\- OpenRouter for LLM API access



\## Contact



For questions or issues, please open a GitHub issue or contact \[your-email@example.com]

