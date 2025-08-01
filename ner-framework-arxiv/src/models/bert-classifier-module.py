"""
BERT Classifier Module
Binary classification model for identifying relevant sentences
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm


class BERTClass(nn.Module):
    """BERT-based binary classifier"""
    
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(BERTClass, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model)
        config.update({'output_hidden_states': True})
        self.l1 = AutoModel.from_pretrained(pretrained_model, config=config)
        self.l2 = nn.Dropout(0.2)
        self.l3 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_pooler = output_1['pooler_output']
        output_2 = self.l2(output_pooler)
        output = self.l3(output_2)
        return output


class CustomDataset(Dataset):
    """Custom dataset for BERT input"""
    
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = dataframe
        self.comment_text = dataframe.comment_text.values.tolist()
        self.max_len = max_len
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, index):
        if isinstance(index, list):
            return [self._get_single_item(i) for i in index]
        else:
            return self._get_single_item(index)
    
    def _get_single_item(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'ids': inputs['input_ids'].squeeze(0),
            'mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs["token_type_ids"].squeeze(0)
        }


class BertClassifier:
    """BERT Classifier wrapper class"""
    
    def __init__(self, model_path, device='cuda', max_len=180):
        self.device = device
        self.max_len = max_len
        self.pretrained_model = 'bert-base-uncased'
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        
        # Initialize and load model
        self.model = BERTClass(self.pretrained_model)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
    
    def predict(self, sentences):
        """
        Predict binary labels for sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            pd.DataFrame: DataFrame with predictions and confidence scores
        """
        # Create dataframe
        df = pd.DataFrame({
            "text": sentences,
            "Bert_binary_labels": [None] * len(sentences)
        })
        df['words'] = df['text'].apply(lambda x: x.split())
        df = df.rename(columns={'text': 'Transcript'})
        
        # Create dataset and dataloader
        new_df = pd.DataFrame({'comment_text': df['Transcript'].values.tolist()})
        dataset = CustomDataset(new_df, self.tokenizer, self.max_len)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=1, drop_last=False)
        
        # Get predictions
        outputs, max_confidences, predicted_labels = self._validate(dataloader)
        
        # Create results dataframe
        df_preds = pd.DataFrame({
            'model_prediction_binary': predicted_labels,
            'model_confidence_score_binary': max_confidences
        })
        
        # Combine with original data
        df_combined = pd.concat([
            df.rename(columns={'Transcript': 'text'}),
            df_preds
        ], axis=1)
        
        # Filter only positive predictions
        df_positive = df_combined[df_combined['model_prediction_binary'] == 1]
        df_positive = df_positive.reset_index(drop=True)
        
        return df_positive[['text', 'words', 'model_prediction_binary', 
                           'model_confidence_score_binary']]
    
    def _validate(self, dataloader):
        """Run validation on dataloader"""
        self.model.eval()
        fin_outputs = []
        sigmoid_v = nn.Sigmoid()
        
        with torch.no_grad():
            for _, data in enumerate(dataloader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                
                ids = ids.squeeze()
                mask = mask.squeeze()
                token_type_ids = token_type_ids.squeeze()
                
                outputs = self.model(ids.unsqueeze(0), mask.unsqueeze(0), 
                                   token_type_ids.unsqueeze(0))
                fin_outputs.extend(sigmoid_v(outputs).cpu().detach().numpy().tolist())
        
        outputs = np.asarray(fin_outputs)
        max_confidences = [max(row) for row in outputs]
        predicted_labels = np.argmax(outputs, axis=1).tolist()
        
        return outputs, max_confidences, predicted_labels