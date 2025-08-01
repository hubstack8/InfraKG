"""
NER Model Module
SciBERT-based Named Entity Recognition for scientific entities
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AutoModelForTokenClassification
import pandas as pd
from tqdm import tqdm


class NERDataset(Dataset):
    """Dataset for NER task"""
    
    def __init__(self, df, tokenizer, max_length=264):
        self.tokenized = tokenizer(
            df['words'].tolist(),
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        self.word_ids_list = [self.tokenized.word_ids(i) for i in range(len(df))]
        self.words = df['words'].tolist()
        self.texts = df['text'].tolist()
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.tokenized['input_ids'][idx],
            'attention_mask': self.tokenized['attention_mask'][idx],
            'word_ids': self.word_ids_list[idx],
            'words': self.words[idx],
            'text': self.texts[idx]
        }
        return item


def custom_collate_fn(batch):
    """Custom collate function for NER dataloader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    word_ids = [item['word_ids'] for item in batch]
    words = [item['words'] for item in batch]
    texts = [item['text'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_ids': word_ids,
        'words': words,
        'text': texts
    }


class NERModel:
    """SciBERT-based NER model wrapper"""
    
    def __init__(self, model_path, model_name='allenai/scibert_scivocab_uncased', device='cuda'):
        """
        Initialize NER model
        
        Args:
            model_path (str): Path to fine-tuned model
            model_name (str): Base model name
            device (str): Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label map
        self.label_map = self.model.config.id2label
    
    def predict(self, df):
        """
        Run NER predictions on dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' and 'words' columns
            
        Returns:
            tuple: (low_confidence_df, full_predictions_df)
        """
        # Create dataset and dataloader
        dataset = NERDataset(df, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
        
        # Inference loop
        all_words_output = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running NER"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                
                for i in range(len(input_ids)):
                    word_ids = batch["word_ids"][i]
                    words = batch["words"][i]
                    text = batch["text"][i]
                    preds = predictions[i]
                    prob_scores = probs[i]
                    last_word_idx = None
                    
                    for token_idx, word_idx in enumerate(word_ids):
                        if word_idx is None or word_idx == last_word_idx:
                            continue
                        last_word_idx = word_idx
                        
                        word = words[word_idx]
                        label_id = preds[token_idx]
                        label = self.label_map[label_id]
                        confidence = prob_scores[token_idx][label_id]
                        
                        all_words_output.append({
                            "text": text,
                            "word": word,
                            "label": label,
                            "confidence_score": confidence
                        })
        
        # Create DataFrame with all words
        df_full_predictions = pd.DataFrame(all_words_output)
        
        # Filter low confidence predictions
        low_confidence_df = df_full_predictions[df_full_predictions['confidence_score'] < 0.70]
        low_confidence_df_no_duplicates = low_confidence_df.drop_duplicates(
            subset=['text'], keep='first'
        )
        
        return low_confidence_df_no_duplicates, df_full_predictions
    
    def extract_entities(self, df_predictions):
        """
        Extract complete entities from word-level predictions
        
        Args:
            df_predictions (pd.DataFrame): DataFrame with word-level predictions
            
        Returns:
            list: List of entity dictionaries
        """
        entities = []
        grouped = df_predictions.groupby('text')
        
        for text, group in grouped:
            current_entity = []
            current_label = None
            current_confidences = []
            
            for _, row in group.iterrows():
                word = row['word']
                label = row['label']
                confidence = row['confidence_score']
                
                if label.startswith("B-"):
                    # Save previous entity if exists
                    if current_entity:
                        entities.append({
                            "text": text,
                            "entity": " ".join(current_entity),
                            "label": current_label,
                            "confidence_score": sum(current_confidences) / len(current_confidences)
                        })
                    # Start new entity
                    current_entity = [word]
                    current_label = label[2:]  # Remove "B-"
                    current_confidences = [confidence]
                
                elif label.startswith("I-") and current_label == label[2:]:
                    # Continue entity
                    current_entity.append(word)
                    current_confidences.append(confidence)
                
                else:
                    # End of entity
                    if current_entity:
                        entities.append({
                            "text": text,
                            "entity": " ".join(current_entity),
                            "label": current_label,
                            "confidence_score": sum(current_confidences) / len(current_confidences)
                        })
                        current_entity = []
                        current_label = None
                        current_confidences = []
            
            # Final entity at end of group
            if current_entity:
                entities.append({
                    "text": text,
                    "entity": " ".join(current_entity),
                    "label": current_label,
                    "confidence_score": sum(current_confidences) / len(current_confidences)
                })
        
        return entities