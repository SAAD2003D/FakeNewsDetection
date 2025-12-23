import sys
import os
import torch
import pandas as pd
from transformers import AutoTokenizer

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Points to 'liar_isot' folder
liar_isot_path = os.path.join(os.path.dirname(current_dir), 'liar_isot')
sys.path.append(liar_isot_path)
sys.path.append(current_dir)

from model_fusion import FusionHCATModel
from dataset_fusion import FusionDataset
from llm_engine import LLMVerifier

# --- CONFIG ---
MODEL_PATH = "fusion_512_epoch_2.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# THRESHOLD: 75% (As requested)
CONFIDENCE_THRESHOLD = 75.0 
MIN_WORDS = 5

def load_system():
    print("--- LOADING HYBRID SYSTEM (Fusion + Groq) ---")
    
    # 1. Initialize LLM (Groq)
    try:
        llm = LLMVerifier()
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize LLM. Error: {e}")
        llm = None
    
    # 2. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # 3. Load Mappings
    data_path = os.path.join(liar_isot_path, "data", "train.tsv")
    if not os.path.exists(data_path):
        print(f"❌ Error: Cannot find training data at {data_path}")
        exit()
        
    temp_ds = FusionDataset(data_path, None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    # 4. Load Model Architecture
    model = FusionHCATModel(
        num_speakers=len(mappings['speaker']), 
        num_parties=len(mappings['party']), 
        num_states=len(mappings['state']), 
        num_subjects=len(mappings['subject'])
    )
    
    # 5. Load Weights (Smart Search)
    weights_path = ""
    if os.path.exists(MODEL_PATH):
        weights_path = MODEL_PATH
    elif os.path.exists(os.path.join(liar_isot_path, MODEL_PATH)):
        weights_path = os.path.join(liar_isot_path, MODEL_PATH)
    else:
        print(f"❌ Error: Could not find model file '{MODEL_PATH}'")
        print("Please check the filename in CONFIG.")
        exit()

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 6. Load Speaker History
    print("Loading Speaker History...")
    df = pd.read_csv(data_path, sep='\t', header=None, on_bad_lines='skip', quoting=3)
    speaker_stats = {}
    for spk in df[4].unique():
        # Get most recent history row
        row = df[df[4] == spk].iloc[-1]
        try:
            raw = [float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])]
            total = sum(raw)
            # Normalize to 0-1
            if total > 0:
                stats = [x / total for x in raw]
            else:
                stats = [0, 0, 0, 0, 0]
        except:
            stats = [0, 0, 0, 0, 0]
        speaker_stats[str(spk)] = stats
        
    return tokenizer, model, mappings, speaker_stats, llm

# ... imports and setup remain the same ...

def main():
    tokenizer, model, mappings, speaker_stats, llm = load_system()
    
    print("\n" + "="*60)
    print("   AI EXPERT DETECTOR (Fusion + Llama 3.3 Expert)")
    print("="*60)
    
    while True:
        text = input("\nClaim: ")
        if text.lower() in ['exit', 'quit']: break
        
        speaker = input("Speaker (Optional): ").strip()
        if speaker.lower() == "trump": speaker = "donald-trump"
        if speaker.lower() == "obama": speaker = "barack-obama"
        party = input("Party (Optional): ").strip().lower()
        
        # --- PHASE 1: FUSION MODEL ---
        s_id = mappings['speaker'].get(speaker, 0)
        p_id = mappings['party'].get(party, 0)
        
        if s_id != 0 and speaker in speaker_stats:
            print(f"  [Context] Using history profile for: {speaker}")
            history = speaker_stats[speaker]
        else:
            if speaker: print(f"  [Context] Speaker unknown. Analyzing Text Style only.")
            history = [0, 0, 0, 0, 0] 

        # Tokenize
        encoding = tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        spk_t = torch.tensor([s_id]).to(DEVICE)
        pty_t = torch.tensor([p_id]).to(DEVICE)
        stt_t = torch.tensor([0]).to(DEVICE)
        sbj_t = torch.tensor([0]).to(DEVICE)
        hist_t = torch.tensor([history], dtype=torch.float).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, spk_t, pty_t, stt_t, sbj_t, hist_t)
            probs = torch.softmax(outputs, dim=1) # [Fake_Prob, Real_Prob]
            pred_class = torch.argmax(probs, dim=1).item()
            
        res = "REAL" if pred_class == 1 else "FAKE"
        conf = probs[0][pred_class].item()
        
        # Store detailed results for the LLM
        hcat_result = {
            "prediction": res,
            "confidence": conf,
            "probabilities": {
                "FAKE": probs[0][0].item(),
                "REAL": probs[0][1].item()
            }
        }
        
        print(f"\n[Fusion Model] Verdict: {res} (Confidence: {conf*100:.2f}%)")
        
        # --- PHASE 2: EXPERT LLM CHECK ---
        word_count = len(text.split())
        needs_llm = False
        reason_trigger = ""
        
        if word_count < MIN_WORDS:
            needs_llm = True
            reason_trigger = "Input too short (Context Missing)"
            # For short text, we might NOT pass hcat_result to force the "Short Fact Check" prompt
            hcat_result = None 
        elif (conf * 100) < CONFIDENCE_THRESHOLD:
            needs_llm = True
            reason_trigger = f"Low Confidence (<{CONFIDENCE_THRESHOLD}%)"
            
        if needs_llm and llm:
            print(f"⚠️  TRIGGERING EXPERT ANALYSIS: {reason_trigger}")
            print("   (Consulting Llama 3.3 Expert Fact-Checker...)")
            
            # Pass the HCAT context to the LLM!
            llm_res, llm_conf, llm_reason = llm.verify(text, hcat_result=hcat_result)
            
            print("-" * 60)
            print(f"🕵️  [EXPERT VERDICT]: {llm_res}")
            print(f"    Confidence: {llm_conf}%")
            print(f"    📝 Analyse:\n    {llm_reason.replace(chr(10), chr(10)+'    ')}") # Indent newlines
            print("-" * 60)
        else:
            print("✅ [System] Accepted Fusion Model result.")

if __name__ == "__main__":
    main()
    
