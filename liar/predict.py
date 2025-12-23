import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from dataset import LiarDataset
from model import HCATModel

MODEL_PATH = "best_hcat_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

def load_system():
    print("Loading Dictionary and Speaker Stats...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Load dataset to get Mappings AND History Stats
    df = pd.read_csv("data/train.tsv", sep='\t', header=None, on_bad_lines='skip', quoting=3)
    # columns 2=text, 4=speaker, 8-12=history
    
    # Create a lookup for speaker history: { 'barack-obama': [1, 0, 5, 10, 0] }
    speaker_stats = {}
    unique_speakers = df[4].unique()
    
    for spk in unique_speakers:
        # Find the most recent row for this speaker
        row = df[df[4] == spk].iloc[0]
        # Columns 8 to 12 are the counts
        try:
            stats = [
                float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])
            ]
        except:
            stats = [0, 0, 0, 0, 0] # Fallback
        speaker_stats[str(spk)] = stats

    # Get Mappings from Dataset Class logic
    temp_ds = LiarDataset("data/train.tsv", tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    print("Loading Model...")
    model = HCATModel(
        num_speakers=len(mappings['speaker']),
        num_parties=len(mappings['party']),
        num_states=len(mappings['state']),
        num_subjects=len(mappings['subject'])
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model, mappings, speaker_stats

def predict_claim(text, speaker, party, state, subject, tokenizer, model, mappings, speaker_stats):
    # 1. Prepare Text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 2. Get Metadata IDs
    s_id = mappings['speaker'].get(speaker, 0)
    p_id = mappings['party'].get(party, 0)
    st_id = mappings['state'].get(state, 0)
    sb_id = mappings['subject'].get(subject, 0)
    
    # 3. Get REAL History Stats
    # If speaker is known, use their real stats. If unknown, use a "Clean" history [0,0,0,0,0]
    if speaker in speaker_stats:
        real_history = speaker_stats[speaker]
        print(f"  [Info] Using history for '{speaker}': {real_history}")
    else:
        print(f"  [Info] Speaker '{speaker}' unknown. Assuming neutral history.")
        real_history = [0, 0, 0, 0, 0] # Assume innocent until proven guilty

    # 4. Tensors
    spk_t = torch.tensor([s_id]).to(DEVICE)
    pty_t = torch.tensor([p_id]).to(DEVICE)
    stt_t = torch.tensor([st_id]).to(DEVICE)
    sbj_t = torch.tensor([sb_id]).to(DEVICE)
    hist_t = torch.tensor([real_history], dtype=torch.float).to(DEVICE) / 100.0 # Normalize!
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # 5. Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, spk_t, pty_t, stt_t, sbj_t, hist_t)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    result = "REAL" if pred_class == 1 else "FAKE"
    confidence = probs[0][pred_class].item() * 100
    
    return result, confidence

if __name__ == "__main__":
    tokenizer, model, mappings, speaker_stats = load_system()
    
    print("\n" + "="*40)
    print("   FAKE NEWS DETECTOR (FIXED)")
    print("="*40)
    
    while True:
        print("\nEnter a Claim (or type 'exit' to quit):")
        text = input("Claim: ")
        if text.lower() == 'exit': break
        
        # Clean inputs
        speaker = input("Speaker: ").strip() # keeping case sensitive matching from mapping
        party = input("Party: ").strip().lower()
        
        # Try to match the exact string format from training
        # If user types "Obama", try to map to "barack-obama"
        if speaker.lower() == "obama": speaker = "barack-obama"
        if speaker.lower() == "trump": speaker = "donald-trump"
        if speaker.lower() == "clinton": speaker = "hillary-clinton"
        
        state = "unknown"
        subject = "unknown"
        
        res, conf = predict_claim(text, speaker, party, state, subject, tokenizer, model, mappings, speaker_stats)
        
        print("-" * 30)
        print(f"VERDICT: {res}")
        print(f"Confidence: {conf:.2f}%")
        print("-" * 30)