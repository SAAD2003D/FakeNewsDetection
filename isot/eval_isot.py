import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_isot import IsotDataset
from model_fusion import FusionHCATModel

# --- CONFIG ---
MODEL_PATH = "isot_only_epoch_3.pth" # Change to 1, 2, or 3 depending on which is best
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
BATCH_SIZE = 16 # 8GB VRAM can handle Batch 16 for inference (no gradients)

def main():
    print(f"--- EVALUATING ISOT MODEL ON {DEVICE} ---")
    
    # 1. Load Data
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # We load the full dataset for evaluation
    test_ds = IsotDataset(
        true_path="isot/data/True.csv",
        fake_path="isot/data/Fake.csv",
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    # IMPORTANT: We use '1' for all dimensions because ISOT has no metadata
    model = FusionHCATModel(num_speakers=1, num_parties=1, num_states=1, num_subjects=1)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Make sure training finished.")
        return

    model.to(DEVICE)
    model.eval()
    
    # 3. Inference Loop
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move inputs to GPU
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            speaker = batch['speaker'].to(DEVICE)
            party = batch['party'].to(DEVICE)
            state = batch['state'].to(DEVICE)
            subject = batch['subject'].to(DEVICE)
            history = batch['history'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Predict
            outputs = model(input_ids, attention_mask, speaker, party, state, subject, history)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Results
    acc = accuracy_score(all_labels, all_preds)
    print("\n" + "="*30)
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    
    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('ISOT-Only Model Performance')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('isot_confusion_matrix.png')
    print("Matrix saved as isot_confusion_matrix.png")

if __name__ == "__main__":
    main()