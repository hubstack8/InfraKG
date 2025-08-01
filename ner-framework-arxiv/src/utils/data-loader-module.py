"""
Data Loader Utilities
Helper functions for loading and managing data
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_metadata(metadata_path: str) -> Optional[Dict]:
    """
    Load metadata from JSON file
    
    Args:
        metadata_path (str): Path to metadata JSON file
        
    Returns:
        dict: Metadata dictionary or None if error
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        return None


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add words column to dataframe by tokenizing text
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' column
        
    Returns:
        pd.DataFrame: DataFrame with added 'words' column
    """
    df['words'] = df['text'].apply(lambda x: x.split())
    return df


def create_sentence_dataframe(sentences: List[str]) -> pd.DataFrame:
    """
    Create DataFrame from list of sentences
    
    Args:
        sentences (List[str]): List of sentences
        
    Returns:
        pd.DataFrame: DataFrame with text and placeholder labels
    """
    df = pd.DataFrame({
        "text": sentences,
        "Bert_binary_labels": [None] * len(sentences)
    })
    return df


def get_pdf_metadata_pairs(folder_path: str) -> List[Tuple[str, str]]:
    """
    Get list of PDF and metadata file pairs from folder
    
    Args:
        folder_path (str): Path to folder containing PDFs and metadata
        
    Returns:
        List[Tuple[str, str]]: List of (pdf_path, metadata_path) tuples
    """
    pairs = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            json_filename = filename.replace('.pdf', '.json')
            json_path = os.path.join(folder_path, json_filename)
            
            if os.path.exists(json_path):
                pairs.append((pdf_path, json_path))
            else:
                logger.warning(f"Missing metadata for: {filename}")
    
    return pairs


def save_results(results: Dict, output_path: str) -> bool:
    """
    Save extraction results to JSON file
    
    Args:
        results (Dict): Extraction results
        output_path (str): Path to save JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        return False


def load_results(results_folder: str) -> List[Dict]:
    """
    Load all result JSON files from folder
    
    Args:
        results_folder (str): Folder containing result JSON files
        
    Returns:
        List[Dict]: List of result dictionaries
    """
    results = []
    
    if not os.path.exists(results_folder):
        logger.warning(f"Results folder not found: {results_folder}")
        return results
    
    for filename in os.listdir(results_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(results_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
    
    return results


def combine_entity_results(results_list: List[Dict]) -> pd.DataFrame:
    """
    Combine entity extraction results from multiple papers
    
    Args:
        results_list (List[Dict]): List of result dictionaries
        
    Returns:
        pd.DataFrame: Combined entity results
    """
    all_entities = []
    
    for result in results_list:
        paper_id = list(result.keys())[0]
        paper_data = result[paper_id]
        
        # Add SciBERT NER results
        for entity in paper_data.get('scibert_ner_results', []):
            entity['paper_id'] = paper_id
            entity['source'] = 'scibert'
            all_entities.append(entity)
        
        # Add LLM results
        for llm_result in paper_data.get('LLMs_ner_results', []):
            if isinstance(llm_result.get('LLMS_NER'), dict):
                for entity_type, entities in llm_result['LLMS_NER'].items():
                    for entity_name in entities:
                        all_entities.append({
                            'paper_id': paper_id,
                            'text': llm_result['sentence'],
                            'entity': entity_name,
                            'label': entity_type,
                            'source': 'llm',
                            'confidence_score': None
                        })
    
    return pd.DataFrame(all_entities)


def filter_entities_by_type(df: pd.DataFrame, entity_types: List[str]) -> pd.DataFrame:
    """
    Filter entities by specific types
    
    Args:
        df (pd.DataFrame): DataFrame with entity results
        entity_types (List[str]): List of entity types to filter
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    return df[df['label'].isin(entity_types)]


def get_entity_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about extracted entities
    
    Args:
        df (pd.DataFrame): DataFrame with entity results
        
    Returns:
        Dict: Statistics dictionary
    """
    stats = {
        'total_entities': len(df),
        'unique_entities': df['entity'].nunique(),
        'papers_processed': df['paper_id'].nunique(),
        'entity_type_counts': df['label'].value_counts().to_dict(),
        'source_counts': df['source'].value_counts().to_dict()
    }
    
    # Add confidence statistics for SciBERT results
    scibert_df = df[df['source'] == 'scibert']
    if len(scibert_df) > 0:
        stats['confidence_stats'] = {
            'mean': scibert_df['confidence_score'].mean(),
            'std': scibert_df['confidence_score'].std(),
            'min': scibert_df['confidence_score'].min(),
            'max': scibert_df['confidence_score'].max()
        }
    
    return stats