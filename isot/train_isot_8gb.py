import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast 

from dataset_isot import IsotDataset
from model_fusion import FusionHCATModel

# --- RTX 4060 (8GB) CONFIGURATION ---
BATCH_SIZE = 8           # 8GB VRAM allows Batch 8 with Mixed Precision
ACCUMULATION_STEPS = 4   # 8 * 4 = 32 Effective Batch Size
MAX_LEN = 512            # Full Text context
EPOCHS = 3               # 3 Epochs is optimal for RoBERTa
LEARNING_RATE = 2e-5     # Standard RoBERTa learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, criterion, scheduler, scaler):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, leave=True)
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(loop):
        # Move data to GPU
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(DEVICE)
        
        # Mixed Precision Forward
        with autocast():
            outputs = model(inputs['input_ids'], inputs['attention_mask'], 
                            inputs['speaker'], inputs['party'], inputs['state'], 
                            inputs['subject'], inputs['history'])
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS

        # Backward
        scaler.scale(loss).backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * ACCUMULATION_STEPS
        loop.set_description(f"Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")

    return total_loss / len(dataloader)

def main():
    print(f"--- TRAINING ISOT-ONLY ON {DEVICE} (8GB Optimizations) ---")
    
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # 1. Load Data
    full_ds = IsotDataset(
        true_path="isot/data/True.csv",
        fake_path="isot/data/Fake.csv",
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # 2. Split into Train (90%) and Val (10%)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    print(f"Train Size: {len(train_ds)} | Val Size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Setup Model
    # Since we have no metadata lists, we pass 1 (dimension size) for all embeddings
    # The embeddings will just learn a bias for "Unknown"
    model = FusionHCATModel(num_speakers=1, num_parties=1, num_states=1, num_subjects=1)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
        print(f"Avg Train Loss: {loss:.4f}")
        
        # Save Model
        save_name = f"isot_only_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Saved: {save_name}")

if __name__ == "__main__":
    main()