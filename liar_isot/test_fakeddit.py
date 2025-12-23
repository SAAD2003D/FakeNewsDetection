import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from model_fusion import FusionHCATModel
from dataset_fusion import FusionDataset

# --- CONFIG ---
MODEL_FILE = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
TEST_SAMPLES = 5000 # Test on 5,000 real articles

def main():
    print(f"--- REALITY CHECK: TESTING ON {TEST_SAMPLES} REAL NEWS ARTICLES ---")
    
    # 1. Load Model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    temp_ds = FusionDataset("liar_isot/data/train.tsv", None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    model = FusionHCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(DEVICE)
    model.eval()
    
    # 2. Load CNN/DailyMail (Guaranteed Real News)
    print("Streaming CNN/DailyMail dataset...")
    # '3.0.0' is the version. We take the 'test' split.
    dataset = load_dataset("cnn_dailymail", '3.0.0', split=f"test[:{TEST_SAMPLES}]")
    
    correct_real = 0
    total = 0
    
    print("Running Inference...")
    for i in tqdm(range(len(dataset))):
        row = dataset[i]
        text = str(row['article']) # Full article text
        
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
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, speaker, party, state, subject, history)
            _, pred = torch.max(outputs, dim=1)
        
        # Label 1 = REAL. We want the model to predict 1.
        if pred.item() == 1:
            correct_real += 1
        total += 1
            
    accuracy = (correct_real / total) * 100
    print("\n" + "="*30)
    print("FALSE POSITIVE STRESS TEST")
    print("="*30)
    print(f"Total Real Articles: {total}")
    print(f"Correctly Identified as Real: {correct_real}")
    print(f"Incorrectly Flagged as Fake: {total - correct_real}")
    print(f"ACCURACY ON REAL NEWS: {accuracy:.2f}%")

if __name__ == "__main__":
    main()