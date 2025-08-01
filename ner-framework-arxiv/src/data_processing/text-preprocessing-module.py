"""
Text Preprocessing Module
Handles text cleaning, sentence splitting, and normalization
"""

import re
import nltk
from nltk.tokenize import sent_tokenize
import unicodedata
from difflib import SequenceMatcher

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass


def clean_text(text):
    """
    Remove unwanted elements from text
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    # Remove <span> elements with page IDs
    text = re.sub(r'<span id="page-\d+-\d+"></span>', '', text)
    
    # Remove LaTeX math blocks between $$ ... $$
    text = re.sub(r'\$\$(.*?)\$\$', '', text, flags=re.DOTALL)
    
    # Remove image syntax ![](...)
    text = re.sub(r'!\[\]\(.*?\)', '', text)
    
    # Remove <sup>...</sup> tags (superscript)
    text = re.sub(r'<sup>.*?</sup>', '', text)
    
    # Remove HTML table tags
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    
    # Remove Markdown-style tables (| column1 | column2 | ...)
    text = re.sub(r'(\|.*\|(?:\n|\r\n?))+', '', text)
    
    # Remove references, URLs, and markdown links entirely
    text = re.sub(r'^\s*-.*\[[^\]]*\]\([^\)]*\).*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-.*$', '', text, flags=re.MULTILINE)
    
    return text


def fun_sentece_split(text):
    """
    Split text into sentences with custom logic
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    # First split using custom regex
    sentence_endings1 = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings1.split(text)
    
    # Process the list to handle long sentences
    final_sentences = []
    for sentence in sentences:
        if len(sentence) > 300:
            # Re-split long sentence using nltk
            split_sentences = sent_tokenize(sentence)
            final_sentences.extend(split_sentences)
        else:
            final_sentences.append(sentence)
    
    # Strip whitespace and remove empty strings
    final_sentences = [s.strip() for s in final_sentences if s.strip()]
    return final_sentences


def split_text_by_abstract(text):
    """
    Split text at Abstract section
    
    Args:
        text (str): Full text
        
    Returns:
        tuple: (text_before_abstract, text_after_abstract)
    """
    match = re.search(r'\bAbstract\b', text, flags=re.IGNORECASE)
    if not match:
        return "", text  # Abstract not found
    before_abstract = text[:match.start()].strip()
    after_abstract = text[match.start():].strip()
    return before_abstract, after_abstract


def normalize_text(text):
    """
    Normalize text for comparison
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Lowercase, strip accents, normalize whitespace
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_match(a, b, threshold=0.8):
    """
    Check if two strings match fuzzily
    
    Args:
        a (str): First string
        b (str): Second string
        threshold (float): Matching threshold
        
    Returns:
        bool: True if strings match above threshold
    """
    return SequenceMatcher(None, a, b).ratio() >= threshold