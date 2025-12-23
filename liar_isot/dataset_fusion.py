import pandas as pd
import torch
from torch.utils.data import Dataset

# LIAR Columns
LIAR_COLS = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 
             'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']

class FusionDataset(Dataset):
    def __init__(self, liar_path, isot_true_path, isot_fake_path, tokenizer, max_len=512, mappings=None, is_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # --- 1. LOAD LIAR DATA (Rich Metadata) ---
        print("Loading LIAR Dataset...")
        df_liar = pd.read_csv(liar_path, sep='\t', names=LIAR_COLS, on_bad_lines='skip', quoting=3)
        df_liar.fillna("unknown", inplace=True)
        
        # Clean History
        hist_cols = ['barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire']
        for col in hist_cols:
            df_liar[col] = pd.to_numeric(df_liar[col], errors='coerce').fillna(0)
            
        # Create or Load Mappings
        if is_train:
            self.speaker_map = self.create_mapping(df_liar['speaker'])
            self.party_map = self.create_mapping(df_liar['party'])
            self.state_map = self.create_mapping(df_liar['state'])
            self.subject_map = self.create_mapping(df_liar['subject'])
        else:
            self.speaker_map = mappings['speaker']
            self.party_map = mappings['party']
            self.state_map = mappings['state']
            self.subject_map = mappings['subject']
            
        liar_labels = {'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1}
        
        for _, row in df_liar.iterrows():
            self.data.append({
                'text': str(row['statement']),
                'speaker': self.speaker_map.get(str(row['speaker']), 0),
                'party': self.party_map.get(str(row['party']), 0),
                'state': self.state_map.get(str(row['state']), 0),
                'subject': self.subject_map.get(str(row['subject']), 0),
                'history': [row[c] for c in hist_cols],
                'label': liar_labels.get(row['label'], 0)
            })

        # --- 2. LOAD ISOT DATA (Full Text) ---
        if isot_true_path and isot_fake_path:
            print("Loading ISOT Dataset (Full Text)...")
            df_true = pd.read_csv(isot_true_path)
            df_fake = pd.read_csv(isot_fake_path)
            
            # Helper to combine Title + Text
            def process_isot(title, text):
                # We put title first, then text. Truncate to avoid huge string processing.
                full_text = str(title) + " " + str(text)
                return full_text[:4000] # Rough char limit before tokenization

            # Load REAL
            for _, row in df_true.iterrows():
                self.data.append({
                    'text': process_isot(row['title'], row['text']),
                    'speaker': 0, 'party': 0, 'state': 0, 'subject': 0, # Unknown Metadata
                    'history': [0,0,0,0,0], # Neutral History
                    'label': 1
                })
                
            # Load FAKE
            for _, row in df_fake.iterrows():
                self.data.append({
                    'text': process_isot(row['title'], row['text']),
                    'speaker': 0, 'party': 0, 'state': 0, 'subject': 0,
                    'history': [0,0,0,0,0],
                    'label': 0
                })
        
        print(f"Total Combined Samples: {len(self.data)}")

    def create_mapping(self, column):
        unique_values = column.astype(str).unique()
        mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        mapping['<UNK>'] = 0 
        return mapping
        
    def get_mappings(self):
        return {'speaker': self.speaker_map, 'party': self.party_map, 'state': self.state_map, 'subject': self.subject_map}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize (The Heavy Lifting)
        encoding = self.tokenizer(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        hist_tensor = torch.tensor(item['history'], dtype=torch.float) / 100.0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'speaker': torch.tensor(item['speaker'], dtype=torch.long),
            'party': torch.tensor(item['party'], dtype=torch.long),
            'state': torch.tensor(item['state'], dtype=torch.long),
            'subject': torch.tensor(item['subject'], dtype=torch.long),
            'history': hist_tensor,
            'label': torch.tensor(item['label'], dtype=torch.long)
        }