import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from dataset_fusion import FusionDataset
from model_fusion import FusionHCATModel

# --- CONFIGURATION ---
# UPDATE THIS TO YOUR SAVED MODEL FILENAME
MODEL_FILE = "fusion_512_epoch_1.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def load_system():
    print("--- LOADING FUSION SYSTEM ---")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # 1. We need LIAR data just to recover the Speaker IDs and History
    print("Loading Metadata Maps...")
    # We use the Dataset class logic to regenerate the mappings
    # We don't need ISOT paths here, just LIAR for the mappings
    temp_ds = FusionDataset(
        liar_path="liar_isot/data/train.tsv", 
        isot_true_path=None, isot_fake_path=None, 
        tokenizer=tokenizer, is_train=True
    )
    mappings = temp_ds.get_mappings()
    
    # 2. Extract History Stats for known speakers
    print("Loading Speaker History...")
    df = pd.read_csv("liar_isot/data/train.tsv", sep='\t', header=None, on_bad_lines='skip', quoting=3)
    speaker_stats = {}
    unique_speakers = df[4].unique()
    for spk in unique_speakers:
        # Get the MOST RECENT history (last row)
        row = df[df[4] == spk].iloc[-1]
        try:
            stats = [float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])]
        except:
            stats = [0, 0, 0, 0, 0]
        speaker_stats[str(spk)] = stats

    # 3. Load Model Architecture
    print(f"Loading Model Weights from {MODEL_FILE}...")
    model = FusionHCATModel(
        num_speakers=len(mappings['speaker']),
        num_parties=len(mappings['party']),
        num_states=len(mappings['state']),
        num_subjects=len(mappings['subject'])
    )
    
    # Load weights (map_location handles CPU/GPU logic automatically)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find file '{MODEL_FILE}'")
        print("Please check the filename in the script!")
        exit()
        
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model, mappings, speaker_stats

def predict(text, speaker, party, tokenizer, model, mappings, speaker_stats):
    # 1. Tokenize (Up to 512 tokens now!)
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 2. Handle Metadata
    # If speaker is empty or unknown, we default to 0 (General News Mode)
    s_id = mappings['speaker'].get(speaker, 0)
    p_id = mappings['party'].get(party, 0)
    
    # Logic: If we know the speaker, use their history.
    # If not, use [0,0,0,0,0] (Neutral/General News)
    if speaker in speaker_stats:
        history = speaker_stats[speaker]
        print(f"  [Context] Using history for politician: {speaker}")
    else:
        history = [0, 0, 0, 0, 0]
        if speaker != "":
            print(f"  [Context] Unknown speaker. Analyzing Text Style only.")

    # 3. Tensors
    spk_t = torch.tensor([s_id]).to(DEVICE)
    pty_t = torch.tensor([p_id]).to(DEVICE)
    stt_t = torch.tensor([0]).to(DEVICE) # Default state
    sbj_t = torch.tensor([0]).to(DEVICE) # Default subject
    hist_t = torch.tensor([history], dtype=torch.float).to(DEVICE) / 100.0
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, spk_t, pty_t, stt_t, sbj_t, hist_t)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    result = "REAL" if pred_class == 1 else "FAKE"
    confidence = probs[0][pred_class].item() * 100
    
    return result, confidence

if __name__ == "__main__":
    tokenizer, model, mappings, speaker_stats = load_system()
    
    print("\n" + "="*50)
    print("   FUSION FAKE NEWS DETECTOR (LIAR + ISOT)")
    print("="*50)
    print("Tip: For general news, leave Speaker/Party empty.")
    
    while True:
        print("\n" + "-"*30)
        text = input("Enter Text: ")
        if text.lower() in ['exit', 'quit']: break
        
        speaker = input("Speaker (Optional): ").strip()
        
        # Auto-fix common names
        if speaker.lower() == "trump": speaker = "donald-trump"
        if speaker.lower() == "obama": speaker = "barack-obama"
        if speaker.lower() == "biden": speaker = "joe-biden"
        
        party = input("Party (Optional): ").strip().lower()
        
        print("\nAnalyzing...")
        res, conf = predict(text, speaker, party, tokenizer, model, mappings, speaker_stats)
        
        print(f"\n>>> VERDICT: {res}")
        print(f">>> Confidence: {conf:.2f}%")