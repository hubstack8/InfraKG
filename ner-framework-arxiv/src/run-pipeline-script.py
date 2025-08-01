"""
Script to run the NER pipeline
"""

import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.main_pipeline import NERPipeline


def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(description='Run NER extraction pipeline')
    
    # Required arguments
    parser.add_argument('--pdf-folder', required=True,
                       help='Folder containing PDF files and metadata')
    parser.add_argument('--output-folder', required=True,
                       help='Folder to save extraction results')
    parser.add_argument('--bert-model', required=True,
                       help='Path to BERT classifier model')
    parser.add_argument('--ner-model', required=True,
                       help='Path to NER model')
    parser.add_argument('--api-key', required=True,
                       help='API key for LLM service')
    
    # Optional arguments
    parser.add_argument('--device', default='cuda',
                       help='Device to run models on (cuda/cpu)')
    parser.add_argument('--log-file', default=None,
                       help='Log file path')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Process files even if output exists')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    if not os.path.exists(args.pdf_folder):
        logger.error(f"PDF folder not found: {args.pdf_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.bert_model):
        logger.error(f"BERT model not found: {args.bert_model}")
        sys.exit(1)
    
    if not os.path.exists(args.ner_model):
        logger.error(f"NER model not found: {args.ner_model}")
        sys.exit(1)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = NERPipeline(
        bert_model_path=args.bert_model,
        ner_model_path=args.ner_model,
        api_key=args.api_key,
        device=args.device
    )
    
    # Process folder
    logger.info(f"Processing PDFs from: {args.pdf_folder}")
    pipeline.process_folder(
        pdf_folder=args.pdf_folder,
        output_folder=args.output_folder,
        skip_existing=not args.no_skip_existing
    )
    
    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()