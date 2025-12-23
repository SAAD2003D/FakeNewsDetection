import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import LiarDataset
from model import HCATModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 4

def main():
    print("--- Generating Final Research Report ---")
    
    # 1. Load Mappings from Train (Essential!)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_ds = LiarDataset("liar/data/train.tsv", tokenizer, is_train=True)
    mappings = train_ds.get_mappings()
    
    # 2. Load TEST Data (Not Validation)
    test_ds = LiarDataset("liar/data/test.tsv", tokenizer, max_len=MAX_LEN, mappings=mappings, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    model = HCATModel(
        num_speakers=len(mappings['speaker']),
        num_parties=len(mappings['party']),
        num_states=len(mappings['state']),
        num_subjects=len(mappings['subject'])
    )
    model.load_state_dict(torch.load("liar/best_hcat_model.pth"))
    model.to(DEVICE)
    model.eval()
    
    # 4. Run Prediction
    all_preds = []
    all_labels = []
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            speaker = batch['speaker'].to(DEVICE)
            party = batch['party'].to(DEVICE)
            state = batch['state'].to(DEVICE)
            subject = batch['subject'].to(DEVICE)
            history = batch['history'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, speaker, party, state, subject, history)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Print Metrics
    print("\n" + "="*30)
    print("FINAL RESULTS (Test Set)")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    
    # 6. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - HCAT Model')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()