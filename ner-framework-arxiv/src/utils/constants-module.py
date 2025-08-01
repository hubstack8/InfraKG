"""
Constants Module
Central location for all constants used in the framework
"""

# Model Constants
BERT_PRETRAINED_MODEL = 'bert-base-uncased'
SCIBERT_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
SENTENCE_TRANSFORMER_MODEL = 'microsoft/mpnet-base'

# Model Parameters
BERT_MAX_LENGTH = 180
NER_MAX_LENGTH = 264
BERT_BATCH_SIZE = 1
NER_BATCH_SIZE = 1
DROPOUT_RATE = 0.2
BERT_HIDDEN_SIZE = 768
NUM_LABELS = 2  # Binary classification

# Thresholds
CONFIDENCE_THRESHOLD = 0.70
SIMILARITY_THRESHOLD = 0.70
FUZZY_MATCH_THRESHOLD = 0.8
SENTENCE_MAX_LENGTH = 300

# LLM Configuration
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "deepseek/deepseek-chat-v3-0324"
LLM_TEMPERATURE = 0.1

# Entity Types
ENTITY_TYPES = {
    'HARDWARE_DEVICE': 'Hardware-device',
    'DEVICE_MEMORY': 'Device-Memory', 
    'DEVICE_COUNT': 'Device-Count',
    'CLOUD_PLATFORM': 'Cloud-Platform',
    'SOFTWARE_ENTITY': 'Software-Entity'
}

# Entity Queries for Embedding Similarity
ENTITY_QUERIES = {
    "Software Entities": [
        "Mention of libraries like TensorFlow, PyTorch, HuggingFace, etc.",
        "Software libraries, tools and framework used in the study",
        "Use of software frameworks such as Keras, OpenCV, Scikit-learn, etc.",
        "Mentions of toolkits or platforms for machine learning and NLP.",
        "Description of the software stack including any libraries or toolkits.",
        "Names of libraries or APIs used in the pipeline.",
        "Which Software libraries used"
    ],
    "Cloud Platform": [
        "Use of cloud computing services for training or deployment.",
        "References to infrastructure providers such as Amazon Web Services or Google Cloud.",
        "Mention of cloud platforms like AWS, GCP, Azure, etc.",
        "Use of cloud resources for scalability or parallel processing.",
        "Utilization of cloud-based solutions in machine learning workflows."
    ],
    "Hardware Device": [
        "GPU and CPU used in the study",
        "Mention of hardware devices such as GPUs, CPUs, or TPUs.",
        "Use of specific hardware like NVIDIA V100, A100, or Intel Xeon processors.",
        "GPU or CPU names referenced in the research",
        "Training performed on high-performance computing devices such as GPUs or TPUs.",
        "Mention of hardware specifications such as number of GPUs or CPUs.",
        "Description of memory usage, such as GPU memory or system RAM.",
        "Mentions of hardware environments including GPU/CPU count and memory size."
    ]
}

# Prompts
ENTITY_EXTRACTION_PROMPT = '''
You are an advanced information extraction system. Your task is to extract entities related to hardware specifications,
cloud platforms, and software tools and libraries and framework from the input text.

Note: Software libraries and tools refer to the tools or libraries used by the authors in their work (e.g., for implementation, training, evaluation, or data processing).
These do not include models or algorithms that are merely cited or evaluated.

Please identify and return entities for the following categories, Extract entities only if they are present in the input text.
Here the Entity Types:

- Hardware-device (GPU and CPU)
- Device-Memory (its a Hardware device memory)
- device Count (its the Total number of Hardware devices used )
- Cloud Platform
- Software Entity


If no entities are found for a given category in the input text, return an empty list for that category.

Return the output in the following JSON format:

{
"Hardware-device": [],
"Device-Memory": [],
"Device-Count": [],
"Cloud-Platform": [],
"Software-Entity": [],

}

'''

AFFILIATION_EXTRACTION_PROMPT = '''
You are a research data normalization and entity classification expert. Your task is to process organization names extracted from research paper affiliations and structure them into the following format:
Instructions:
Main Organization:
  •	Identify and return the main organization name (e.g., university, national lab, research institute, or company).
  •	Normalize its spelling and casing.
Sub-Organizations:
•	Extract and normalize all sub-organizations affiliated with the main organization, including:
  o	Departments
  o	Laboratories
  o	Schools
  o	Institutes
  o	Joint centers
  o	Research centers
  o	Hospitals
  o	Subsidiary companies
  o	Other structural units
•	Normalize abbreviations (e.g., "thuai" → "Tsinghua University Institute for Artificial Intelligence"), spelling, and casing.
Categorization:
•	Categorize each sub-organization into one of the following types:
  o	Department
  o	Institute
  o	Laboratory
  o	School
  o	Company
  o	Joint Center
  o	Hospital
  o	Research Center
  o	Other

Important:
- Only include information actually present in the input text.
- If a certain type of sub-organization or category is not found in the text, do not invent or fill it—leave it empty or omit it.
- However, for the Main Organization, you are allowed to infer or supply the university or main entity name even if it is not explicitly mentioned in the input.

Output Format (JSON):
{
  "Main Organization": "Full Normalized Name of Main Organization",
  "Sub-Organizations": [
    {
      "Name": "Normalized Sub-Organization Name",
      "Category": "Department | Institute | Laboratory | School | Company | Joint Center | Hospital | Research Center | Other"
    },
    ...
  ]
}

Input Text:
'''

# File Patterns
PDF_EXTENSION = '.pdf'
JSON_EXTENSION = '.json'

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Default Paths
DEFAULT_MODELS_DIR = 'models'
DEFAULT_DATA_DIR = 'data'
DEFAULT_OUTPUT_DIR = 'outputs'
DEFAULT_LOG_DIR = 'logs'

# Error Messages
ERROR_MESSAGES = {
    'pdf_not_found': "PDF file not found: {}",
    'metadata_not_found': "Metadata file not found: {}",
    'model_not_found': "Model file not found: {}",
    'api_key_missing': "API key not provided",
    'gpu_not_available': "GPU not available, using CPU",
    'extraction_failed': "Failed to extract entities from: {}",
    'invalid_json': "Invalid JSON format in file: {}"
}

# Success Messages
SUCCESS_MESSAGES = {
    'model_loaded': "Successfully loaded model: {}",
    'processing_complete': "Successfully processed: {}",
    'results_saved': "Results saved to: {}",
    'batch_complete': "Batch processing complete. Processed {} files."
}