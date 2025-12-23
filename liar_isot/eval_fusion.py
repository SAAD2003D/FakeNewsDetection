import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom fusion files
from dataset_fusion import FusionDataset
from model_fusion import FusionHCATModel

# --- CONFIG ---
MODEL_PATH = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
BATCH_SIZE = 8

def main():
    print("--- EVALUATING FUSION MODEL (SMART MODE) ---")
    
    # 1. Load Tokenizer & Mappings from Train
    print("Loading Mappings from Train...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_temp = FusionDataset(
        liar_path="liar_isot/data/train.tsv", 
        isot_true_path=None, isot_fake_path=None, 
        tokenizer=tokenizer, 
        is_train=True
    )
    mappings = train_temp.get_mappings()
    
    # 2. Load Test Data
    print("Loading Test Data...")
    test_ds = FusionDataset(
        liar_path="liar_isot/data/test.tsv", 
        isot_true_path=None, isot_fake_path=None, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN, 
        mappings=mappings, 
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    print("Loading Model Weights...")
    model = FusionHCATModel(
        num_speakers=len(mappings['speaker']),
        num_parties=len(mappings['party']),
        num_states=len(mappings['state']),
        num_subjects=len(mappings['subject'])
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    
    # 4. Run Prediction Loop
    all_preds = []
    all_labels = []
    
    print(f"Running inference on {len(test_ds)} samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move to GPU
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(DEVICE)
            
            # --- SMART LOGIC IMPLEMENTATION ---
            # In predict_smart.py, if speaker is unknown (ID 0), we force history to [0,0,0,0,0].
            # Here, we do the same using Vector Masks for the whole batch.
            
            speaker_ids = inputs['speaker']
            history_tensor = inputs['history']
            
            # Create a mask where Speaker == 0 (Unknown)
            # unsqueeze(1) makes shape [Batch, 1] to broadcast over the history dimension
            unknown_mask = (speaker_ids == 0).unsqueeze(1).expand_as(history_tensor)
            
            # Apply the mask: Wherever speaker is 0, replace history with 0.0
            # This removes the "Unknown Speaker Penalty"
            smart_history = history_tensor.masked_fill(unknown_mask, 0.0)
            
            # ----------------------------------
            
            # Predict using the cleaned history
            outputs = model(inputs['input_ids'], inputs['attention_mask'], 
                            inputs['speaker'], inputs['party'], inputs['state'], 
                            inputs['subject'], smart_history)
            
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Generate Report
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n>>> OVERALL ACCURACY (Smart Mode): {acc*100:.2f}% <<<")
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    
    # 6. Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Smart Fusion Accuracy: {acc*100:.1f}%')
    plt.savefig('fusion_smart_confusion_matrix.png')
    print("Confusion Matrix saved as 'fusion_smart_confusion_matrix.png'")

if __name__ == "__main__":
    main()
    
