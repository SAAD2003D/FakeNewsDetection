import torch
from transformers import AutoTokenizer
from model_fusion import FusionHCATModel

# --- CONFIG ---
MODEL_PATH = "isot_only_epoch_3.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

def load_system():
    print("Loading ISOT Model...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Initialize with size 1 (The training script used size 1 for metadata)
    model = FusionHCATModel(num_speakers=1, num_parties=1, num_states=1, num_subjects=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    
    return tokenizer, model

def predict_text(text, tokenizer, model):
    # 1. Tokenize
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
    
    # 2. Dummy Metadata (Zeros)
    # The ISOT model was trained to ignore these, but the architecture still expects them
    zero_tensor = torch.tensor([0]).to(DEVICE)
    zero_hist = torch.tensor([[0,0,0,0,0]], dtype=torch.float).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_hist)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        
    result = "REAL" if pred_class == 1 else "FAKE"
    confidence = probs[0][pred_class].item() * 100
    return result, confidence

if __name__ == "__main__":
    tokenizer, model = load_system()
    
    print("\n" + "="*50)
    print("   ISOT FAKE NEWS DETECTOR (TEXT ONLY)")
    print("="*50)
    
    while True:
        print("\n" + "-"*30)
        text = input("Enter News Text (or 'exit'): ")
        if text.lower() in ['exit', 'quit']: break
        
        if len(text) < 10:
            print("⚠️ Text too short. Please enter a full sentence or headline.")
            continue
            
        print("Analyzing Style...")
        res, conf = predict_text(text, tokenizer, model)
        
        print(f"\n>>> VERDICT: {res}")
        print(f">>> Confidence: {conf:.2f}%")