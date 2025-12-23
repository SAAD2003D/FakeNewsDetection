import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import LiarDataset
from model import HCATModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

def main():
    print("--- SCANNING FOR 'REAL' PREDICTIONS ---")
    
    # Load Data
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_ds = LiarDataset("data/train.tsv", tokenizer, is_train=True)
    mappings = train_ds.get_mappings()
    
    # Load Test Set
    test_ds = LiarDataset("data/test.tsv", tokenizer, max_len=MAX_LEN, mappings=mappings, is_train=False)
    
    # Load Model
    model = HCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.load_state_dict(torch.load("best_hcat_model.pth"))
    model.to(DEVICE)
    model.eval()
    
    count_real = 0
    
    print("\nHere are claims the model thinks are REAL:\n")
    
    for i in range(len(test_ds)):
        sample = test_ds[i]
        
        # Prepare inputs
        inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample.items() if k != 'label'}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item() * 100
        
        # If model predicts REAL (1) and is confident (>60%)
        if pred == 1 and conf > 60:
            # Decode the text back to words
            text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            print(f"[{count_real+1}] CONFIDENCE: {conf:.2f}%")
            print(f"CLAIM: {text}")
            print("-" * 50)
            count_real += 1
            
        if count_real >= 5: break # Stop after finding 5 examples

if __name__ == "__main__":
    main()