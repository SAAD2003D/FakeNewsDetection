import streamlit as st
import torch
import sys
import os
import pandas as pd
from transformers import AutoTokenizer

# --- PATH SETUP ---
# Allow importing from sibling folders
current_dir = os.path.dirname(os.path.abspath(__file__))
liar_isot_path = os.path.join(os.path.dirname(current_dir), 'liar_isot')
sys.path.append(liar_isot_path)
sys.path.append(current_dir)

from model_fusion import FusionHCATModel
from dataset_fusion import FusionDataset
from llm_engine import LLMVerifier

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FactGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# --- CONFIG ---
MODEL_PATH = "fusion_512_epoch_2.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CACHED LOAD FUNCTION (Crucial for Speed) ---
# This keeps the model in memory so it doesn't reload every time you click a button
@st.cache_resource
def load_system_resources():
    print("Loading resources...")
    
    # 1. LLM
    try:
        llm = LLMVerifier()
    except:
        llm = None
        
    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Mappings
    data_path = os.path.join(liar_isot_path, "data", "train.tsv")
    temp_ds = FusionDataset(data_path, None, None, tokenizer, is_train=True)
    mappings = temp_ds.get_mappings()
    
    model = FusionHCATModel(
        num_speakers=len(mappings['speaker']), 
        num_parties=len(mappings['party']), 
        num_states=len(mappings['state']), 
        num_subjects=len(mappings['subject'])
    )
    
    # Weights
    weights_path = ""
    if os.path.exists(MODEL_PATH): weights_path = MODEL_PATH
    elif os.path.exists(os.path.join(liar_isot_path, MODEL_PATH)): weights_path = os.path.join(liar_isot_path, MODEL_PATH)
    
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Speaker History
    df = pd.read_csv(data_path, sep='\t', header=None, on_bad_lines='skip', quoting=3)
    speaker_stats = {}
    for spk in df[4].unique():
        row = df[df[4] == spk].iloc[-1]
        try:
            raw = [float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])]
            total = sum(raw)
            if total > 0: stats = [x / total for x in raw]
            else: stats = [0]*5
        except: stats = [0]*5
        speaker_stats[str(spk)] = stats
        
    return tokenizer, model, mappings, speaker_stats, llm

# Load resources
tokenizer, model, mappings, speaker_stats, llm = load_system_resources()

# --- UI DESIGN ---
st.title("🛡️ FactGuard AI")
st.markdown("### Hybrid Fake News Detection System")
st.markdown("Combines **My model** (roBERTa based) with **Expert LLM Verification**.")

# Sidebar for Settings
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Expert Intervention Threshold (%)", 50, 100, 75)
    st.info(f"If the Model confidence is below **{threshold}%**, the Expert LLM (Llama 3) will verify the claim.")

# Input Form
with st.form("analysis_form"):
    claim = st.text_area("Enter News Claim:", height=100, placeholder="e.g. The government passed a new tax law yesterday...")
    
    col1, col2 = st.columns(2)
    with col1:
        speaker = st.text_input("Speaker (Optional)", placeholder="e.g. Donald Trump")
    with col2:
        party = st.selectbox("Party (Optional)", ["None", "republican", "democrat", "libertarian", "green"])
        
    submitted = st.form_submit_button("🔍 Analyze Veracity")

# Logic
if submitted and claim:
    # Pre-processing
    if party == "None": party = ""
    clean_speaker = speaker.strip().replace(" ", "-").lower() # Match dataset format
    clean_party = party.strip().lower()
    
    # --- STEP 1: FUSION MODEL ---
    with st.spinner("Analyzing linguistic patterns & political history..."):
        # Metadata logic
        s_id = mappings['speaker'].get(clean_speaker, 0)
        p_id = mappings['party'].get(clean_party, 0)
        
        if s_id != 0 and clean_speaker in speaker_stats:
            history = speaker_stats[clean_speaker]
            context_msg = f"Using history profile for: **{clean_speaker}**"
        else:
            history = [0, 0, 0, 0, 0]
            context_msg = "Unknown speaker. Analyzing **Text Style** only."

        # Inference
        encoding = tokenizer(claim, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        spk_t = torch.tensor([s_id]).to(DEVICE)
        pty_t = torch.tensor([p_id]).to(DEVICE)
        stt_t = torch.tensor([0]).to(DEVICE)
        sbj_t = torch.tensor([0]).to(DEVICE)
        hist_t = torch.tensor([history], dtype=torch.float).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, spk_t, pty_t, stt_t, sbj_t, hist_t)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            
        res = "REAL" if pred_class == 1 else "FAKE"
        conf = probs[0][pred_class].item() * 100
        
        # Prepare HCAT result for LLM
        hcat_result = {
            "prediction": res,
            "confidence": conf / 100.0,
            "probabilities": {"FAKE": probs[0][0].item(), "REAL": probs[0][1].item()}
        }

    # --- STEP 2: DECISION GATE ---
    word_count = len(claim.split())
    needs_expert = False
    
    if word_count < 6:
        reason = "Input too short"
        needs_expert = True
    elif conf < threshold:
        reason = f"Confidence ({conf:.1f}%) < Threshold ({threshold}%)"
        needs_expert = True
        
    # --- DISPLAY RESULTS ---
    
    # 1. Fusion Result
    st.divider()
    st.subheader("🤖 Initial AI Assessment")
    st.caption(context_msg)
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if res == "REAL":
            st.success(f"**{res}**")
        else:
            st.error(f"**{res}**")
    with col_b:
        st.progress(int(conf), text=f"AI Confidence: {conf:.1f}%")

    # 2. Expert Result (Conditional)
    if needs_expert:
        st.warning(f"⚠️ **Expert Verification Triggered** ({reason})")
        
        with st.spinner("Consulting Llama 3 Expert (Groq)..."):
            llm_res, llm_conf, llm_reason = llm.verify(claim, hcat_result)
            
        st.subheader("🕵️ Expert Verdict")
        
        final_color = "green" if llm_res == "REAL" else "red"
        st.markdown(f":{final_color}[**VERDICT: {llm_res}**]")
        st.markdown(f"**Confidence:** {llm_conf:.1f}%")
        
        with st.expander("📝 View Expert Analysis", expanded=True):
            st.write(llm_reason)
    else:
        st.success("✅ AI Confidence is high. No expert verification needed.")