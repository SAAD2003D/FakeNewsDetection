import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision (FP16)

from dataset_fusion import FusionDataset
from model_fusion import FusionHCATModel

# --- 4GB GPU EXTREME CONFIG ---
BATCH_SIZE = 1           # Only 1 sample at a time fits with 512 tokens
ACCUMULATION_STEPS = 32  # Update weights every 32 steps (Effective Batch = 32)
MAX_LEN = 512            # Full Text
EPOCHS = 3               # Dataset is large, 2 epochs is enough
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- STARTING FUSION TRAINING (512 TOKENS) ON {DEVICE} ---")
    
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # 1. Load Data
    train_ds = FusionDataset(
        liar_path="liar_isot/data/train.tsv",
        isot_true_path="liar_isot/data/True.csv",
        isot_fake_path="liar_isot/data/Fake.csv",
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        is_train=True
    )
    
    # Validation on LIAR only (to check generalizing)
    mappings = train_ds.get_mappings()
    valid_ds = FusionDataset("liar_isot/data/valid.tsv", None, None, tokenizer, max_len=MAX_LEN, mappings=mappings, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False) # Val can be slightly larger
    
    # 2. Setup Model
    model = FusionHCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() # For FP16
    
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(loop):
            # Move data
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(DEVICE)
            
            # --- Mixed Precision Forward Pass ---
            with autocast():
                outputs = model(inputs['input_ids'], inputs['attention_mask'], 
                                inputs['speaker'], inputs['party'], inputs['state'], 
                                inputs['subject'], inputs['history'])
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS # Normalize loss

            # --- Backward Pass ---
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update Progress Bar
            current_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss
            if i % 100 == 0: # Only update text occasionally to save speed
                loop.set_description(f"Loss: {current_loss:.4f}")

        # Save Model at end of epoch
        print(f"Epoch Finished. Avg Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"fusion_512_epoch_{epoch+1}.pth")
        print("Model Saved.")

if __name__ == "__main__":
    main()