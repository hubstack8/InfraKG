# API Documentation

## Core Pipeline

### `NERPipeline`

Main pipeline class that orchestrates the entire extraction process.

```python
class NERPipeline(bert_model_path: str, ner_model_path: str, api_key: str, device: str = 'cuda')
```

**Parameters:**
- `bert_model_path` (str): Path to the BERT binary classifier model
- `ner_model_path` (str): Path to the SciBERT NER model
- `api_key` (str): API key for LLM service
- `device` (str): Device to run models on ('cuda' or 'cpu')

**Methods:**

#### `process_pdf(pdf_path: str, metadata_path: str) -> Optional[Dict]`

Process a single PDF file and extract entities.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `metadata_path` (str): Path to the corresponding metadata JSON file

**Returns:**
- Dictionary with paper_id as key and extraction results as value
- None if processing fails

**Example:**
```python
pipeline = NERPipeline(bert_path, ner_path, api_key)
result = pipeline.process_pdf("paper.pdf", "paper.json")
```

#### `process_folder(pdf_folder: str, output_folder: str, skip_existing: bool = True)`

Process all PDFs in a folder.

**Parameters:**
- `pdf_folder` (str): Folder containing PDF files and metadata
- `output_folder` (str): Folder to save extraction results
- `skip_existing` (bool): Skip processing if output already exists

## Data Processing

### `pdf_parser(pdf_path: str) -> str`

Extract text content from a PDF file.

**Parameters:**
- `pdf_path` (str): Path to PDF file

**Returns:**
- Extracted text as string

### `clean_text(text: str) -> str`

Remove unwanted elements from text (LaTeX, HTML tags, etc.).

**Parameters:**
- `text` (str): Raw text to clean

**Returns:**
- Cleaned text

### `fun_sentece_split(text: str) -> List[str]`

Split text into sentences with custom logic.

**Parameters:**
- `text` (str): Text to split

**Returns:**
- List of sentences

### `affliation_extractor(text: str, metadata: dict) -> str`

Extract author affiliations from paper header.

**Parameters:**
- `text` (str): Text before abstract section
- `metadata` (dict): Paper metadata with title and authors

**Returns:**
- Extracted affiliation text

## Models

### `BertClassifier`

Binary classifier for identifying relevant sentences.

```python
class BertClassifier(model_path: str, device: str = 'cuda', max_len: int = 180)
```

**Methods:**

#### `predict(sentences: List[str]) -> pd.DataFrame`

Predict binary labels for sentences.

**Parameters:**
- `sentences` (List[str]): List of sentences to classify

**Returns:**
- DataFrame with predictions and confidence scores

### `EmbeddingSimilarity`

Find sentences similar to entity-related queries.

```python
class EmbeddingSimilarity(model_name: str = 'microsoft/mpnet-base')
```

**Methods:**

#### `find_similar_sentences(sentences: List[str], threshold: float = 0.70) -> pd.DataFrame`

Find sentences similar to predefined entity queries.

**Parameters:**
- `sentences` (List[str]): List of sentences to search
- `threshold` (float): Similarity threshold (0-1)

**Returns:**
- DataFrame with similar sentences

### `NERModel`

SciBERT-based Named Entity Recognition model.

```python
class NERModel(model_path: str, model_name: str = 'allenai/scibert_scivocab_uncased', device: str = 'cuda')
```

**Methods:**

#### `predict(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`

Run NER predictions on sentences.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with 'text' and 'words' columns

**Returns:**
- Tuple of (low_confidence_df, full_predictions_df)

#### `extract_entities(df_predictions: pd.DataFrame) -> List[Dict]`

Extract complete entities from word-level predictions.

**Parameters:**
- `df_predictions` (pd.DataFrame): DataFrame with word-level predictions

**Returns:**
- List of entity dictionaries

## LLM Client

### `LLMClient`

Client for Large Language Model API calls.

```python
class LLMClient(api_key: str, base_url: str = "https://openrouter.ai/api/v1", 
                model: str = "deepseek/deepseek-chat-v3-0324")
```

**Methods:**

#### `extract_entities(low_confidence_df: pd.DataFrame) -> List[Dict]`

Extract entities from low confidence predictions using LLM.

**Parameters:**
- `low_confidence_df` (pd.DataFrame): DataFrame with low confidence sentences

**Returns:**
- List of extraction results with LLM predictions

#### `extract_affiliations(affiliation_text: str) -> List[Dict]`

Extract and normalize affiliations using LLM.

**Parameters:**
- `affiliation_text` (str): Raw affiliation text

**Returns:**
- List with extraction result

## Output Schema

### Main Output Structure

```json
{
  "paper_id": {
    "paper_title": "string",
    "paper_authors": ["string"],
    "paper_publish_year": "string",
    "paper_category": ["string"],
    "paper_url": "string",
    "paper_doi": "string",
    "paper_comments": "string",
    "paper_journal_ref": "string",
    "scibert_ner_results": [EntityResult],
    "LLMs_ner_results": [LLMResult],
    "LLMs_affliations_extraction": AffiliationResult
  }
}
```

### EntityResult Schema

```json
{
  "text": "string",
  "entity": "string",
  "label": "string",
  "confidence_score": "float"
}
```

### LLMResult Schema

```json
{
  "sentence": "string",
  "response_text": "string",
  "LLMS_NER": {
    "Hardware-device": ["string"],
    "Device-Memory": ["string"],
    "Device-Count": ["string"],
    "Cloud-Platform": ["string"],
    "Software-Entity": ["string"]
  }
}
```

### AffiliationResult Schema

```json
{
  "Main Organization": "string",
  "Sub-Organizations": [
    {
      "Name": "string",
      "Category": "string"
    }
  ]
}
```

## Entity Types

### Supported Entity Labels

- **Hardware-device**: GPUs, CPUs, TPUs (e.g., "NVIDIA V100", "Intel Xeon")
- **Device-Memory**: Memory specifications (e.g., "32GB RAM", "16GB GPU memory")
- **Device-Count**: Number of devices (e.g., "8 GPUs", "4 nodes")
- **Cloud-Platform**: Cloud services (e.g., "AWS", "Google Cloud", "Azure")
- **Software-Entity**: Libraries and frameworks (e.g., "TensorFlow", "PyTorch")

## Error Handling

All methods include error handling and logging. Errors are logged with appropriate context.

### Common Exceptions

- `FileNotFoundError`: PDF or metadata file not found
- `json.JSONDecodeError`: Invalid JSON metadata
- `torch.cuda.OutOfMemoryError`: GPU memory exhausted
- `openai.APIError`: LLM API call failed

### Example Error Handling

```python
try:
    result = pipeline.process_pdf(pdf_path, metadata_path)
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Usage

- BERT model: ~440MB
- SciBERT NER model: ~440MB
- Embedding model: ~420MB
- Total GPU memory recommended: 8GB+

### Processing Speed

- With GPU: ~20-30 seconds per paper
- With CPU: ~2-3 minutes per paper
- Batch processing recommended for large datasets

### Optimization Tips

1. Use GPU whenever possible
2. Process files in batches to optimize GPU utilization
3. Enable `skip_existing` to avoid reprocessing
4. Adjust confidence thresholds based on precision/recall needs