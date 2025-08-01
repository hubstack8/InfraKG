"""
Embedding Similarity Module
Uses sentence embeddings to find relevant sentences based on entity queries
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, models, util
import torch


class EmbeddingSimilarity:
    """Embedding similarity search for entity-related sentences"""
    
    def __init__(self, model_name='microsoft/mpnet-base'):
        """
        Initialize embedding model
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        # Load SentenceTransformer model using CLS pooling
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False,
            pooling_mode_max_tokens=False
        )
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # Define entity query sets
        self.entity_queries = {
            "Software Entities": [
                "Mention of libraries like TensorFlow, PyTorch, HuggingFace, etc.",
                "Software libraries, tools and framework used in the study",
                "Use of software frameworks such as Keras, OpenCV, Scikit-learn, etc.",
                "Mentions of toolkits or platforms for machine learning and NLP.",
                "Description of the software stack including any libraries or toolkits.",
                "Names of libraries or APIs used in the pipeline.",
                "Which Software libraries used"
            ],
            "Cloud Platform": [
                "Use of cloud computing services for training or deployment.",
                "References to infrastructure providers such as Amazon Web Services or Google Cloud.",
                "Mention of cloud platforms like AWS, GCP, Azure, etc.",
                "Use of cloud resources for scalability or parallel processing.",
                "Utilization of cloud-based solutions in machine learning workflows."
            ],
            "Hardware Device": [
                "GPU and CPU used in the study",
                "Mention of hardware devices such as GPUs, CPUs, or TPUs.",
                "Use of specific hardware like NVIDIA V100, A100, or Intel Xeon processors.",
                "GPU or CPU names referenced in the research",
                "Training performed on high-performance computing devices such as GPUs or TPUs.",
                "Mention of hardware specifications such as number of GPUs or CPUs.",
                "Description of memory usage, such as GPU memory or system RAM.",
                "Mentions of hardware environments including GPU/CPU count and memory size."
            ]
        }
    
    def find_similar_sentences(self, sentences, threshold=0.70):
        """
        Find sentences similar to entity queries
        
        Args:
            sentences (list): List of sentences to search
            threshold (float): Similarity threshold
            
        Returns:
            pd.DataFrame: DataFrame with similar sentences
        """
        # Create dataframe
        df = pd.DataFrame({
            "text": sentences,
            "words": [s.split() for s in sentences]
        })
        
        # Encode sentences
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # List to collect relevant results
        results = []
        
        # Process each entity and query
        for entity_name, queries in self.entity_queries.items():
            for query in queries:
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
                
                for idx, score in enumerate(cos_scores):
                    if score >= threshold:
                        results.append({
                            "text": sentences[idx],
                            "entity name": entity_name,
                            "similarity score": float(score),
                            "words": df.loc[idx, "words"],
                        })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            return pd.DataFrame(columns=['text', 'words'])
        
        # Remove duplicates
        results_df['normalized_text'] = results_df['text'].str.lower().str.strip()
        results_df = results_df.drop_duplicates(subset='normalized_text', keep='first')
        results_df = results_df.drop(columns='normalized_text').reset_index(drop=True)
        
        return results_df[['text', 'words']]
    
    def combine_with_bert_results(self, similarity_results, bert_results):
        """
        Combine similarity results with BERT classifier results
        
        Args:
            similarity_results (pd.DataFrame): Results from similarity search
            bert_results (pd.DataFrame): Results from BERT classifier
            
        Returns:
            pd.DataFrame: Combined unique results