"""
Simple example of using the NER Framework
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NERPipeline


def main():
    # Configuration
    BERT_MODEL_PATH = "models/final_binarymodel.pt"
    NER_MODEL_PATH = "models/scibert_NER_final_model"
    API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")
    
    # Check if models exist
    if not os.path.exists(BERT_MODEL_PATH):
        print(f"Error: BERT model not found at {BERT_MODEL_PATH}")
        print("Please download the model first.")
        return
    
    if not os.path.exists(NER_MODEL_PATH):
        print(f"Error: NER model not found at {NER_MODEL_PATH}")
        print("Please download the model first.")
        return
    
    # Initialize pipeline
    print("Initializing NER pipeline...")
    pipeline = NERPipeline(
        bert_model_path=BERT_MODEL_PATH,
        ner_model_path=NER_MODEL_PATH,
        api_key=API_KEY,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example 1: Process a single paper
    print("\n" + "="*50)
    print("Example 1: Processing a single paper")
    print("="*50)
    
    pdf_path = "sample_data/example_paper.pdf"
    metadata_path = "sample_data/example_paper.json"
    
    if os.path.exists(pdf_path) and os.path.exists(metadata_path):
        result = pipeline.process_pdf(pdf_path, metadata_path)
        
        if result:
            paper_id = list(result.keys())[0]
            paper_data = result[paper_id]
            
            print(f"\nPaper: {paper_data['paper_title']}")
            print(f"Authors: {', '.join(paper_data['paper_authors'][:3])}...")
            
            # Show some extracted entities
            entities = paper_data['scibert_ner_results'][:5]
            print(f"\nSample entities found:")
            for entity in entities:
                print(f"  - {entity['entity']} ({entity['label']}): {entity['confidence_score']:.2f}")
            
            # Save result
            output_file = "example_output.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nFull results saved to: {output_file}")
    else:
        print("Sample files not found. Please add example_paper.pdf and example_paper.json to sample_data/")
    
    # Example 2: Batch processing
    print("\n" + "="*50)
    print("Example 2: Batch processing")
    print("="*50)
    
    pdf_folder = "sample_data/papers"
    output_folder = "sample_data/results"
    
    if os.path.exists(pdf_folder):
        print(f"\nProcessing all PDFs in: {pdf_folder}")
        pipeline.process_folder(
            pdf_folder=pdf_folder,
            output_folder=output_folder,
            skip_existing=True
        )
        print(f"Results saved to: {output_folder}")
    else:
        print(f"Folder not found: {pdf_folder}")
        print("Create this folder and add PDF files with corresponding JSON metadata.")
    
    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)


if __name__ == "__main__":
    import torch
    main()