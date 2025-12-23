import pandas as pd
import torch
from torch.utils.data import Dataset

class IsotDataset(Dataset):
    def __init__(self, true_path, fake_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print("Loading ISOT Dataset...")
        
        # 1. Load Real News
        df_true = pd.read_csv(true_path)
        for _, row in df_true.iterrows():
            text = str(row['title']) + " " + str(row['text'])
            self.data.append({
                'text': text[:5000], # Truncate raw text to save tokenization time
                'label': 1 # REAL
            })
            
        # 2. Load Fake News
        df_fake = pd.read_csv(fake_path)
        for _, row in df_fake.iterrows():
            text = str(row['title']) + " " + str(row['text'])
            self.data.append({
                'text': text[:5000],
                'label': 0 # FAKE
            })
            
        print(f"Total ISOT Samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize (Full 512 context)
        encoding = self.tokenizer(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Hardcode Metadata to 0 (Unknown) for ISOT
        # This keeps the Fusion Architecture happy
        zero_meta = torch.tensor(0, dtype=torch.long)
        zero_hist = torch.tensor([0,0,0,0,0], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'speaker': zero_meta,
            'party': zero_meta,
            'state': zero_meta,
            'subject': zero_meta,
            'history': zero_hist,
            'label': torch.tensor(item['label'], dtype=torch.long)
        }