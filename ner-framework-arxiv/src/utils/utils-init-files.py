# src/__init__.py
"""NER Framework for Scientific Literature Analysis"""

__version__ = "1.0.0"
__author__ = "Your Name"

# src/data_processing/__init__.py
"""Data processing modules"""

from .pdf_parser import pdf_parser, json_file_reading
from .text_preprocessing import (
    clean_text, fun_sentece_split, split_text_by_abstract,
    normalize_text, fuzzy_match
)
from .affiliation_extractor import affliation_extractor

__all__ = [
    'pdf_parser', 'json_file_reading', 'clean_text', 
    'fun_sentece_split', 'split_text_by_abstract',
    'normalize_text', 'fuzzy_match', 'affliation_extractor'
]

# src/models/__init__.py
"""Model modules"""

from .bert_classifier import BertClassifier, BERTClass
from .embedding_similarity import EmbeddingSimilarity
from .ner_model import NERModel

__all__ = ['BertClassifier', 'BERTClass', 'EmbeddingSimilarity', 'NERModel']

# src/api/__init__.py
"""API client modules"""

from .llm_client import LLMClient

__all__ = ['LLMClient']

# src/pipeline/__init__.py
"""Pipeline modules"""

from .main_pipeline import NERPipeline

__all__ = ['NERPipeline']

# src/utils/__init__.py
"""Utility modules"""

from .data_loader import (
    load_metadata, load_data, create_sentence_dataframe,
    get_pdf_metadata_pairs, save_results, load_results,
    combine_entity_results, filter_entities_by_type,
    get_entity_statistics
)

from .constants import (
    BERT_PRETRAINED_MODEL, SCIBERT_MODEL_NAME,
    BERT_MAX_LENGTH, NER_MAX_LENGTH,
    CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD,
    ENTITY_TYPES, ENTITY_QUERIES,
    ENTITY_EXTRACTION_PROMPT, AFFILIATION_EXTRACTION_PROMPT,
    ERROR_MESSAGES, SUCCESS_MESSAGES
)

__all__ = [
    # data_loader functions
    'load_metadata', 'load_data', 'create_sentence_dataframe',
    'get_pdf_metadata_pairs', 'save_results', 'load_results',
    'combine_entity_results', 'filter_entities_by_type',
    'get_entity_statistics',
    # constants
    'BERT_PRETRAINED_MODEL', 'SCIBERT_MODEL_NAME',
    'BERT_MAX_LENGTH', 'NER_MAX_LENGTH',
    'CONFIDENCE_THRESHOLD', 'SIMILARITY_THRESHOLD',
    'ENTITY_TYPES', 'ENTITY_QUERIES',
    'ENTITY_EXTRACTION_PROMPT', 'AFFILIATION_EXTRACTION_PROMPT',
    'ERROR_MESSAGES', 'SUCCESS_MESSAGES'
]