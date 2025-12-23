import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report
from model_fusion import FusionHCATModel
from dataset_fusion import FusionDataset

# --- CONFIG ---
EXTERNAL_CSV = "liar_isot/data/covid/covid_test.csv" # <--- NEW DATASET
MODEL_FILE = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def main():
    print("--- TESTING ON UNSEEN COVID-19 DATA ---")
    
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    temp_ds = FusionDataset("liar_isot/data/train.tsv", None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    model = FusionHCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(DEVICE)
    model.eval()
    
    print(f"Reading {EXTERNAL_CSV}...")
    df = pd.read_csv(EXTERNAL_CSV)
    
    preds = []
    labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            
            # 1. Handle COVID Dataset Columns ('tweet', 'label')
            text = str(row['tweet'])
            label_str = str(row['label']).lower()
            
            # Convert 'real'/'fake' text to 1/0
            if 'real' in label_str:
                true_label = 1
            elif 'fake' in label_str:
                true_label = 0
            else:
                continue # Skip weird rows

            # 2. Tokenize
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 3. Feed "Unknown" Metadata (Model relies on Text Style only)
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)
            
            speaker = torch.tensor([0]).to(DEVICE)
            party = torch.tensor([0]).to(DEVICE)
            state = torch.tensor([0]).to(DEVICE)
            subject = torch.tensor([0]).to(DEVICE)
            history = torch.tensor([[0,0,0,0,0]], dtype=torch.float).to(DEVICE)
            
            # 4. Predict
            outputs = model(input_ids, attention_mask, speaker, party, state, subject, history)
            _, pred = torch.max(outputs, dim=1)
            
            preds.append(pred.item())
            labels.append(true_label)
            
    print("\n" + "="*30)
    print("COVID-19 DOMAIN TRANSFER RESULTS")
    print("="*30)
    print(classification_report(labels, preds, target_names=['Fake', 'Real']))

if __name__ == "__main__":
    main()