"""
PDF Parser Module
Handles PDF text extraction and metadata reading
"""

import fitz  # PyMuPDF
import json
import logging

logger = logging.getLogger(__name__)


def pdf_parser(pdf_path):
    """
    Extract text from PDF file
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {e}")
        return None


def json_file_reading(file_path_meta):
    """
    Read metadata from JSON file
    
    Args:
        file_path_meta (str): Path to JSON metadata file
        
    Returns:
        dict: Parsed JSON data or None if error
    """
    try:
        with open(file_path_meta, 'r', encoding='utf-8') as file:
            meta_json = json.load(file)
        return meta_json
    except Exception as e:
        logger.error(f"Error reading JSON: {file_path_meta} - {e}")
        return None