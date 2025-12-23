import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from model_fusion import FusionHCATModel
from dataset_fusion import FusionDataset

# --- CONFIG ---
DATA_DIR = "liar_isot/data/training/training/fakeNewsDataset" # Folder containing 'fake' and 'legit' folders
MODEL_FILE = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""

def load_amt_data():
    data = []
    
    # 1. Load FAKE (Label 0)
    fake_dir = os.path.join(DATA_DIR, "fake")
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(".txt"):
                # Extract domain from filename (e.g., "biz01fake.txt" -> "Business")
                domain = "Unknown"
                if "biz" in filename: domain = "Business"
                elif "edu" in filename: domain = "Education"
                elif "ent" in filename: domain = "Entertainment"
                elif "pol" in filename: domain = "Politics"
                elif "spo" in filename: domain = "Sports"
                elif "tech" in filename: domain = "Technology"
                
                content = read_text_file(os.path.join(fake_dir, filename))
                if content:
                    data.append({'text': content, 'label': 0, 'domain': domain})

    # 2. Load LEGIT (Label 1)
    legit_dir = os.path.join(DATA_DIR, "legit")
    if os.path.exists(legit_dir):
        for filename in os.listdir(legit_dir):
            if filename.endswith(".txt"):
                # Extract domain
                domain = "Unknown"
                if "biz" in filename: domain = "Business"
                elif "edu" in filename: domain = "Education"
                elif "ent" in filename: domain = "Entertainment"
                elif "pol" in filename: domain = "Politics"
                elif "spo" in filename: domain = "Sports"
                elif "tech" in filename: domain = "Technology"
                
                content = read_text_file(os.path.join(legit_dir, filename))
                if content:
                    data.append({'text': content, 'label': 1, 'domain': domain})
    
    return pd.DataFrame(data)

def main():
    print("--- TESTING ON MULTI-DOMAIN DATASET (FakeNewsAMT) ---")
    
    # 1. Load Model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    temp_ds = FusionDataset("liar_isot/data/train.tsv", None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    model = FusionHCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(DEVICE)
    model.eval()
    
    # 2. Load Data
    df = load_amt_data()
    print(f"Loaded {len(df)} samples across {df['domain'].nunique()} domains.")
    
    # 3. Predict
    all_preds = []
    all_labels = []
    
    # We will store results per domain to show breakdown
    domain_results = {d: {'preds': [], 'labels': []} for d in df['domain'].unique()}
    
    print("Running Inference...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            domain = row['domain']
            label = row['label']
            
            # Tokenize
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            # Unknown Metadata
            speaker = torch.tensor([0]).to(DEVICE)
            party = torch.tensor([0]).to(DEVICE)
            state = torch.tensor([0]).to(DEVICE)
            subject = torch.tensor([0]).to(DEVICE)
            history = torch.tensor([[0,0,0,0,0]], dtype=torch.float).to(DEVICE)
            
            # Predict
            outputs = model(input_ids, attention_mask, speaker, party, state, subject, history)
            _, pred = torch.max(outputs, dim=1)
            
            p_val = pred.item()
            
            # Store Global
            all_preds.append(p_val)
            all_labels.append(label)
            
            # Store Domain Specific
            domain_results[domain]['preds'].append(p_val)
            domain_results[domain]['labels'].append(label)

    # 4. Print Breakdown
    print("\n" + "="*40)
    print("RESULTS BY DOMAIN")
    print("="*40)
    print(f"{'DOMAIN':<15} | {'ACCURACY':<10} | {'SAMPLES'}")
    print("-" * 40)
    
    for domain, data in domain_results.items():
        acc = accuracy_score(data['labels'], data['preds'])
        print(f"{domain:<15} | {acc*100:.2f}%     | {len(data['labels'])}")
        
    print("-" * 40)
    print(f"\nOVERALL ACCURACY: {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

if __name__ == "__main__":
    main()