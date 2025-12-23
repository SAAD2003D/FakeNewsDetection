import torch
import pandas as pd
from transformers import AutoTokenizer
from dataset_fusion import FusionDataset
from model_fusion import FusionHCATModel

# --- CONFIG ---
MODEL_FILE = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def load_system():
    print("--- LOADING SMART FUSION SYSTEM ---")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Load Mappings to get correct dimensions
    temp_ds = FusionDataset("liar_isot/data/train.tsv", None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    model = FusionHCATModel(len(mappings['speaker']), len(mappings['party']), len(mappings['state']), len(mappings['subject']))
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load Real History
    df = pd.read_csv("liar_isot/data/train.tsv", sep='\t', header=None, on_bad_lines='skip', quoting=3)
    speaker_stats = {}
    for spk in df[4].unique():
        row = df[df[4] == spk].iloc[-1]
        try:
            # Normalize stats immediately (0 to 1 range)
            raw_stats = [float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])]
            total = sum(raw_stats)
            if total > 0:
                stats = [x / total for x in raw_stats] # Turn counts into percentages
            else:
                stats = [0, 0, 0, 0, 0]
        except:
            stats = [0, 0, 0, 0, 0]
        speaker_stats[str(spk)] = stats
        
    return tokenizer, model, mappings, speaker_stats

def predict(text, speaker, party, tokenizer, model, mappings, speaker_stats):
    encoding = tokenizer(text, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    
    # --- SMART LOGIC ---
    s_id = mappings['speaker'].get(speaker, 0)
    p_id = mappings['party'].get(party, 0)
    
    # If speaker is found, use their REAL history
    if s_id != 0 and speaker in speaker_stats:
        print(f"  [Context] Found known politician: {speaker}")
        history = speaker_stats[speaker]
    else:
        # CRITICAL FIX: If speaker is unknown, use a "Perfect Neutral" history
        # Instead of [0,0,0,0,0], let's try to mimic a standard ISOT article
        print(f"  [Context] Unknown speaker. Analyzing Text Style only.")
        history = [0, 0, 0, 0, 0] 

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Tensors
    spk_t = torch.tensor([s_id]).to(DEVICE)
    pty_t = torch.tensor([p_id]).to(DEVICE)
    stt_t = torch.tensor([0]).to(DEVICE)
    sbj_t = torch.tensor([0]).to(DEVICE)
    hist_t = torch.tensor([history], dtype=torch.float).to(DEVICE) # Already normalized
    
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
    print("   SMART PREDICTOR")
    print("="*50)
    
    while True:
        text = input("\nClaim: ")
        if text.lower() == 'exit': break
        
        # Force user to type a longer sentence
        if len(text.split()) < 5:
            print("⚠️  Warning: Text is very short. Model might guess FAKE due to lack of context.")
            
        speaker = input("Speaker (Optional): ").strip()
        if speaker.lower() == "trump": speaker = "donald-trump"
        if speaker.lower() == "obama": speaker = "barack-obama"
        
        party = input("Party (Optional): ").strip().lower()
        
        res, conf = predict(text, speaker, party, tokenizer, model, mappings, speaker_stats)
        print(f"\n>>> VERDICT: {res} ({conf:.2f}%)")