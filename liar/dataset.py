import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 1. Define the Column Names for LIAR
COLUMNS = [
    'id',                
    'label',             
    'statement',         
    'subject',           
    'speaker',           
    'job_title',         
    'state_info',        
    'party_affiliation', 
    'barely_true_counts', # History Count 1
    'false_counts',      # History Count 2
    'half_true_counts',  # History Count 3
    'mostly_true_counts',# History Count 4
    'pants_on_fire_counts', # History Count 5
    'context'            
]

class LiarDataset(Dataset):
    def __init__(self, tsv_file, tokenizer, max_len=128, mappings=None, is_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load the data
        # 'quoting=3' helps handle messy quotes in the text
        self.df = pd.read_csv(tsv_file, sep='\t', names=COLUMNS, on_bad_lines='skip', quoting=3)
        
        # --- CRITICAL FIX: Clean the History Columns ---
        # Some rows have text in these columns due to parsing errors.
        # We force them to be numbers. If it's text, it becomes NaN, then 0.
        history_cols = [
            'barely_true_counts', 'false_counts', 'half_true_counts', 
            'mostly_true_counts', 'pants_on_fire_counts'
        ]
        
        for col in history_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Handle other Missing Values
        self.df.fillna("unknown", inplace=True)

        # --- Metadata Logic ---
        if is_train:
            self.speaker_map = self.create_mapping(self.df['speaker'])
            self.party_map = self.create_mapping(self.df['party_affiliation'])
            self.state_map = self.create_mapping(self.df['state_info'])
            self.subject_map = self.create_mapping(self.df['subject'])
            
            self.label_map = {
                'pants-fire': 0, 
                'false': 0, 
                'barely-true': 0, 
                'half-true': 1, 
                'mostly-true': 1, 
                'true': 1
            }
        else:
            self.speaker_map = mappings['speaker']
            self.party_map = mappings['party']
            self.state_map = mappings['state']
            self.subject_map = mappings['subject']
            self.label_map = mappings['label']

    def create_mapping(self, column):
        # Convert to string to ensure we don't crash on mixed types
        unique_values = column.astype(str).unique()
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        mapping['<UNK>'] = 0 
        return mapping

    def get_mappings(self):
        return {
            'speaker': self.speaker_map,
            'party': self.party_map,
            'state': self.state_map,
            'subject': self.subject_map,
            'label': self.label_map
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Process Text
        text = str(row['statement']) # Force string
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 2. Process Metadata (Use str() to be safe)
        speaker_id = self.speaker_map.get(str(row['speaker']), 0)
        party_id = self.party_map.get(str(row['party_affiliation']), 0)
        state_id = self.state_map.get(str(row['state_info']), 0)
        subject_id = self.subject_map.get(str(row['subject']), 0)
        
        # 3. Process History (Now guaranteed to be float)
        history = torch.tensor([
            row['barely_true_counts'], row['false_counts'],
            row['half_true_counts'], row['mostly_true_counts'],
            row['pants_on_fire_counts']
        ], dtype=torch.float) / 100.0

        # 4. Process Label
        label_id = self.label_map.get(row['label'], 1)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'speaker': torch.tensor(speaker_id, dtype=torch.long),
            'party': torch.tensor(party_id, dtype=torch.long),
            'state': torch.tensor(state_id, dtype=torch.long),
            'subject': torch.tensor(subject_id, dtype=torch.long),
            'history': history,
            'label': torch.tensor(label_id, dtype=torch.long)
        }