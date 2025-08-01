"""
Main Pipeline Module
Orchestrates the complete NER extraction pipeline
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional

from ..data_processing.pdf_parser import pdf_parser, json_file_reading
from ..data_processing.text_preprocessing import (
    clean_text, fun_sentece_split, split_text_by_abstract
)
from ..data_processing.affiliation_extractor import affliation_extractor
from ..models.bert_classifier import BertClassifier
from ..models.embedding_similarity import EmbeddingSimilarity
from ..models.ner_model import NERModel
from ..api.llm_client import LLMClient

logger = logging.getLogger(__name__)


class NERPipeline:
    """Main pipeline for NER extraction from scientific papers"""
    
    def __init__(self, bert_model_path: str, ner_model_path: str, 
                 api_key: str, device: str = 'cuda'):
        """
        Initialize the NER pipeline
        
        Args:
            bert_model_path: Path to BERT classifier model
            ner_model_path: Path to NER model
            api_key: API key for LLM service
            device: Device to run models on
        """
        self.device = device
        
        # Initialize models
        logger.info("Initializing BERT classifier...")
        self.bert_classifier = BertClassifier(bert_model_path, device=device)
        
        logger.info("Initializing embedding similarity model...")
        self.embedding_similarity = EmbeddingSimilarity()
        
        logger.info("Initializing NER model...")
        self.ner_model = NERModel(ner_model_path, device=device)
        
        logger.info("Initializing LLM client...")
        self.llm_client = LLMClient(api_key)
    
    def process_pdf(self, pdf_path: str, metadata_path: str) -> Optional[Dict]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            metadata_path: Path to metadata JSON file
            
        Returns:
            Dictionary with extraction results or None if error
        """
        try:
            # Read metadata
            metadata = json_file_reading(metadata_path)
            if not metadata:
                logger.error(f"Cannot read metadata: {metadata_path}")
                return None
            
            paper_id = metadata.get("id")
            if not paper_id:
                logger.error(f"No paper ID in metadata: {metadata_path}")
                return None
            
            # Extract text from PDF
            logger.info(f"Processing paper: {paper_id}")
            pdf_text = pdf_parser(pdf_path)
            if not pdf_text:
                logger.error(f"Failed to extract text from PDF: {pdf_path}")
                return None
            
            # Split text
            text_affiliation, text_without_aff = split_text_by_abstract(pdf_text)
            
            # Process affiliations
            logger.info("Extracting affiliations...")
            cleaned_affiliation = affliation_extractor(text_affiliation, metadata)
            affiliation_results = self.llm_client.extract_affiliations(cleaned_affiliation)
            
            # Preprocess text
            clean_text_without_aff = clean_text(text_without_aff)
            clean_sentences = fun_sentece_split(clean_text_without_aff)
            logger.info(f"Split into {len(clean_sentences)} sentences")
            
            # BERT classification
            logger.info("Running BERT classification...")
            bert_results = self.bert_classifier.predict(clean_sentences)
            logger.info(f"Found {len(bert_results)} relevant sentences")
            
            # Embedding similarity
            logger.info("Running embedding similarity...")
            similarity_results = self.embedding_similarity.find_similar_sentences(
                clean_sentences
            )
            combined_sentences = self.embedding_similarity.combine_with_bert_results(
                similarity_results, bert_results
            )
            logger.info(f"Combined to {len(combined_sentences)} unique sentences")
            
            # NER extraction
            logger.info("Running NER model...")
            low_confidence_df, full_predictions = self.ner_model.predict(combined_sentences)
            entities = self.ner_model.extract_entities(full_predictions)
            
            # LLM enhancement for low confidence
            logger.info("Enhancing with LLM...")
            llm_results = self.llm_client.extract_entities(low_confidence_df)
            
            # Construct result
            result = {
                "paper_title": metadata.get("title", ""),
                "paper_authors": metadata.get("authors", []),
                "paper_publish_year": metadata.get("published", ""),
                "paper_category": metadata.get("categories", []),
                "paper_url": metadata.get("links", {}).get("alternate", ""),
                "paper_doi": metadata.get("doi", ""),
                "paper_comments": metadata.get("comment", ""),
                "paper_journal_ref": metadata.get("journal_ref", ""),
                "scibert_ner_results": entities,
                "LLMs_ner_results": llm_results,
                "LLMs_affliations_extraction": affiliation_results[0]['LLMS_Affiliation']
            }
            
            return {paper_id: result}
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return None
    
    def process_folder(self, pdf_folder: str, output_folder: str, 
                      skip_existing: bool = True):
        """
        Process all PDFs in a folder
        
        Args:
            pdf_folder: Folder containing PDFs and metadata
            output_folder: Folder to save results
            skip_existing: Skip if output already exists
        """
        os.makedirs(output_folder, exist_ok=True)
        
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for filename in pdf_files:
            pdf_path = os.path.join(pdf_folder, filename)
            json_filename = filename.replace('.pdf', '.json')
            json_path = os.path.join(pdf_folder, json_filename)
            
            if not os.path.exists(json_path):
                logger.warning(f"Missing metadata for: {filename}")
                continue
            
            # Check if should skip
            metadata = json_file_reading(json_path)
            if metadata and skip_existing:
                paper_id = metadata.get('id')
                if paper_id:
                    output_path = os.path.join(output_folder, f"{paper_id}.json")
                    if os.path.exists(output_path):
                        logger.info(f"Skipping existing: {paper_id}")
                        continue
            
            # Process PDF
            result = self.process_pdf(pdf_path, json_path)
            
            if result:
                paper_id = list(result.keys())[0]
                output_path = os.path.join(output_folder, f"{paper_id}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4)
                
                logger.info(f"Saved results for: {paper_id}")
            else:
                logger.error(f"Failed to process: {filename}")