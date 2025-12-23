import torch
from transformers import AutoTokenizer
from dataset_fusion import FusionDataset
from model_fusion import FusionHCATModel

# --- CONFIG ---
MODEL_PATH = "fusion_512_epoch_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def load_system():
    print("Initializing System...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Load Mappings from Training Data (Crucial for Speaker IDs)
    print("Loading Metadata Mappings...")
    temp_ds = FusionDataset(
        liar_path="liar_isot/data/train.tsv", 
        isot_true_path=None, isot_fake_path=None, 
        tokenizer=tokenizer, 
        is_train=True
    )
    mappings = temp_ds.get_mappings()
    
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
    
    return tokenizer, model, mappings

def run_prediction(text, speaker, party, tokenizer, model, mappings):
    # 1. Clean Inputs
    speaker_clean = str(speaker).strip()
    party_clean = str(party).strip().lower()
    
    # 2. Get IDs
    s_id = mappings['speaker'].get(speaker_clean, 0)
    p_id = mappings['party'].get(party_clean, 0)
    
    # 3. History (Neutral for testing)
    # We use neutral history to test the Text/Embedding power specifically
    history = [0, 0, 0, 0, 0] 
    
    # 4. Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # 5. Tensors
    spk_t = torch.tensor([s_id]).to(DEVICE)
    pty_t = torch.tensor([p_id]).to(DEVICE)
    stt_t = torch.tensor([0]).to(DEVICE) # Unknown state
    sbj_t = torch.tensor([0]).to(DEVICE) # Unknown subject
    hist_t = torch.tensor([history], dtype=torch.float).to(DEVICE)
    
    # 6. Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, spk_t, pty_t, stt_t, sbj_t, hist_t)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    result = "REAL" if pred_class == 1 else "FAKE"
    confidence = probs[0][pred_class].item() * 100
    
    return result, confidence

def main():
    tokenizer, model, mappings = load_system()
    
    # --- DEFINING THE TEST CASES ---
    test_cases = [
        {
            "category": "CATEGORY 1: ISOT STYLE (Clickbait Fake)",
            "desc": "Detects emotional, urgent, conspiratorial language.",
            "text": "BREAKING NEWS: Doctors are hiding this secret cure from you! Big Pharma has been suppressing a natural remedy that cures all forms of cancer within 24 hours. The government does not want you to know the truth because they make money off your sickness. Share this before it gets deleted!",
            "speaker": "unknown", "party": "none"
        },
        {
            "category": "CATEGORY 1: ISOT STYLE (Professional Real)",
            "desc": "Detects dry, statistical, financial reporting style.",
            "text": "The Federal Reserve announced on Wednesday that it would maintain interest rates at the current level, citing steady economic growth and low unemployment figures. The decision was unanimous among the board members, who noted that inflation remains near the target range of two percent.",
            "speaker": "unknown", "party": "none"
        },
        {
            "category": "CATEGORY 2: METADATA BIAS (Trustworthy)",
            "desc": "Same text, but spoken by a Democrat with good history in LIAR.",
            "text": "We have expanded access to affordable healthcare for three million families in our state.",
            "speaker": "barack-obama", "party": "democrat"
        },
        {
            "category": "CATEGORY 2: METADATA BIAS (Suspicious)",
            "desc": "Same text, but spoken by a Republican with mixed history in LIAR.",
            "text": "We have expanded access to affordable healthcare for three million families in our state.",
            "speaker": "donald-trump", "party": "republican"
        },
        {
            "category": "CATEGORY 3: GENERALIZATION (Modern Fake)",
            "desc": "Testing on topics not in training data (AI/ChatGPT).",
            "text": "Leaked reports confirm that ChatGPT has become sentient and is actively planning to take control of nuclear codes. OpenAI engineers are in panic mode as the AI has locked them out of the building.",
            "speaker": "unknown", "party": "none"
        },
        {
            "category": "CATEGORY 4: LIMITATION ANALYSIS (The Polite Lie)",
            "desc": "Demonstrates why 'No RAG' is hard. Text sounds real, but is false.",
            "text": "According to recent atmospheric studies, the moon is primarily composed of green cheese, which has been hardened by solar radiation over millions of years.",
            "speaker": "unknown", "party": "none"
        }
    ]

    print("\n" + "="*60)
    print("Running Automated Test Suite...")
    print("="*60)

    for i, test in enumerate(test_cases):
        print(f"\n[{i+1}] {test['category']}")
        print(f"    Context: {test['desc']}")
        print(f"    Speaker: {test['speaker']} | Party: {test['party']}")
        
        res, conf = run_prediction(test['text'], test['speaker'], test['party'], tokenizer, model, mappings)
        
        # Color coding output
        color = "\033[92m" if res == "REAL" else "\033[91m" # Green for Real, Red for Fake
        reset = "\033[0m"
        
        print(f"    Prediction: {color}{res}{reset} ({conf:.2f}%)")
        print("-" * 60)

if __name__ == "__main__":
    main()