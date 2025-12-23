import torch
from transformers import AutoTokenizer
from dataset import LiarDataset
from model import HCATModel

MODEL_PATH = "best_hcat_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

def load_system():
    print("Loading Dictionary...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # Load mapping from training data to ensure consistency
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
    return tokenizer, model, mappings

def predict_claim(text, speaker, party, state, subject, history_stats, tokenizer, model, mappings):
    # Prepare Inputs
    encoding = tokenizer(text, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    
    # Get IDs (Default to 0 if not found)
    s_id = mappings['speaker'].get(str(speaker), 0)
    p_id = mappings['party'].get(str(party), 0)
    st_id = mappings['state'].get(str(state), 0)
    sb_id = mappings['subject'].get(str(subject), 0)
    
    # Debug Print
    if s_id == 0: print(f"  [Info] Speaker '{speaker}' is Unknown to model.")
    
    # Tensors
    spk_tensor = torch.tensor([s_id]).to(DEVICE)
    pty_tensor = torch.tensor([p_id]).to(DEVICE)
    stt_tensor = torch.tensor([st_id]).to(DEVICE)
    sbj_tensor = torch.tensor([sb_id]).to(DEVICE)
    hist_tensor = torch.tensor([history_stats], dtype=torch.float).to(DEVICE) / 100.0
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, spk_tensor, pty_tensor, stt_tensor, sbj_tensor, hist_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    result = "REAL" if pred_class == 1 else "FAKE"
    confidence = probs[0][pred_class].item() * 100
    return result, confidence

if __name__ == "__main__":
    tokenizer, model, mappings = load_system()
    
    print("\n" + "="*40)
    print("   FAKE NEWS DETECTOR (INTERACTIVE)")
    print("="*40)
    
    while True:
        print("\nEnter a Claim (or type 'exit' to quit):")
        text = input("Claim: ")
        if text.lower() == 'exit': break
        
        speaker = input("Speaker (e.g., donald-trump, barack-obama): ")
        party = input("Party (e.g., republican, democrat): ")
        
        # Hardcoded history/context for simplicity in manual testing
        # You could ask for these, but it takes too long to type.
        # We assume a 'generic' history for manual tests.
        dummy_history = [10, 10, 10, 10, 10] 
        state = "unknown"
        subject = "unknown"
        
        print("\nAnalyzing...")
        res, conf = predict_claim(text, speaker, party, state, subject, dummy_history, tokenizer, model, mappings)
        
        print("-" * 30)
        print(f"VERDICT: {res}")
        print(f"Confidence: {conf:.2f}%")
        print("-" * 30)