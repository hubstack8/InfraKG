"""
LLM Client Module
Handles API calls to Large Language Models for entity extraction
"""

import json
import re
import logging
from openai import OpenAI
import pandas as pd

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM API calls"""
    
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1", 
                 model="deepseek/deepseek-chat-v3-0324"):
        """
        Initialize LLM client
        
        Args:
            api_key (str): API key for authentication
            base_url (str): Base URL for API
            model (str): Model identifier
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        
        # Define prompts
        self.entity_prompt = '''
You are an advanced information extraction system. Your task is to extract entities related to hardware specifications,
cloud platforms, and software tools and libraries and framework from the input text.

Note: Software libraries and tools refer to the tools or libraries used by the authors in their work (e.g., for implementation, training, evaluation, or data processing).
These do not include models or algorithms that are merely cited or evaluated.

Please identify and return entities for the following categories, Extract entities only if they are present in the input text.
Here the Entity Types:

- Hardware-device (GPU and CPU)
- Device-Memory (its a Hardware device memory)
- device Count (its the Total number of Hardware devices used )
- Cloud Platform
- Software Entity


If no entities are found for a given category in the input text, return an empty list for that category.

Return the output in the following JSON format:

{
"Hardware-device": [],
"Device-Memory": [],
"Device-Count": [],
"Cloud-Platform": [],
"Software-Entity": [],

}

'''
        
        self.affiliation_prompt = '''
You are a research data normalization and entity classification expert. Your task is to process organization names extracted from research paper affiliations and structure them into the following format:
Instructions:
Main Organization:
  •	Identify and return the main organization name (e.g., university, national lab, research institute, or company).
  •	Normalize its spelling and casing.
Sub-Organizations:
•	Extract and normalize all sub-organizations affiliated with the main organization, including:
  o	Departments
  o	Laboratories
  o	Schools
  o	Institutes
  o	Joint centers
  o	Research centers
  o	Hospitals
  o	Subsidiary companies
  o	Other structural units
•	Normalize abbreviations (e.g., "thuai" → "Tsinghua University Institute for Artificial Intelligence"), spelling, and casing.
Categorization:
•	Categorize each sub-organization into one of the following types:
  o	Department
  o	Institute
  o	Laboratory
  o	School
  o	Company
  o	Joint Center
  o	Hospital
  o	Research Center
  o	Other

Important:
- Only include information actually present in the input text.
- If a certain type of sub-organization or category is not found in the text, do not invent or fill it—leave it empty or omit it.
- However, for the Main Organization, you are allowed to infer or supply the university or main entity name even if it is not explicitly mentioned in the input.

Output Format (JSON):
{
  "Main Organization": "Full Normalized Name of Main Organization",
  "Sub-Organizations": [
    {
      "Name": "Normalized Sub-Organization Name",
      "Category": "Department | Institute | Laboratory | School | Company | Joint Center | Hospital | Research Center | Other"
    },
    ...
  ]
}

Input Text:
'''
    
    def _parse_json_output(self, text):
        """
        Parse JSON from LLM response
        
        Args:
            text (str): LLM response text
            
        Returns:
            dict or str: Parsed JSON or error message
        """
        # Check for JSON code block
        code_block_match = re.search(r'```json(.*?)```', text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            try:
                obj, end = decoder.raw_decode(text[idx:])
                return obj
            except json.JSONDecodeError:
                idx += 1
        
        return 'None response was empty'
    
    def _call_api(self, prompt, input_text, temperature=0.1):
        """
        Make API call to LLM
        
        Args:
            prompt (str): System prompt
            input_text (str): User input
            temperature (float): Sampling temperature
            
        Returns:
            str: LLM response
        """
        prompt_text = prompt + " " + input_text
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            temperature=temperature
        )
        
        return completion.choices[0].message.content
    
    def extract_entities(self, low_confidence_df):
        """
        Extract entities from low confidence predictions using LLM
        
        Args:
            low_confidence_df (pd.DataFrame): DataFrame with low confidence sentences
            
        Returns:
            list: List of extraction results
        """
        results = []
        
        for i, row in low_confidence_df.iterrows():
            sentence = row['text'].replace('\n', ' ').replace('\r', ' ')
            try:
                response_text = self._call_api(self.entity_prompt, sentence)
                parsed_response = self._parse_json_output(response_text)
            except Exception as e:
                logger.error(f"Error in entry {i}: {e}")
                response_text = "ERROR"
                parsed_response = []
            
            results.append({
                "sentence": sentence,
                "response_text": response_text,
                "LLMS_NER": parsed_response
            })
        
        return results
    
    def extract_affiliations(self, affiliation_text):
        """
        Extract and normalize affiliations using LLM
        
        Args:
            affiliation_text (str): Raw affiliation text
            
        Returns:
            list: List with extraction result
        """
        results = []
        
        try:
            response_text = self._call_api(self.affiliation_prompt, affiliation_text)
            parsed_response = self._parse_json_output(response_text)
        except Exception as e:
            logger.error(f"Error in affiliation extraction: {e}")
            response_text = "ERROR"
            parsed_response = []
        
        results.append({
            "sentence": affiliation_text,
            "response_text": response_text,
            "LLMS_Affiliation": parsed_response
        })
        
        return results