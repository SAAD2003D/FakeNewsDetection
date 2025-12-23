import torch
import torch.nn as nn
from transformers import AutoModel

class FusionHCATModel(nn.Module):
    def __init__(self, num_speakers, num_parties, num_states, num_subjects):
        super(FusionHCATModel, self).__init__()
        
        print("Loading RoBERTa (Memory Optimized)...")
        self.roberta = AutoModel.from_pretrained('roberta-base')
        
        # --- MEMORY HACK FOR 4GB GPU ---
        # This trades speed for memory. It re-computes parts of the graph 
        # during backprop instead of storing them. Essential for 512 tokens.
        self.roberta.gradient_checkpointing_enable()
        
        # Metadata Embeddings
        self.speaker_emb = nn.Embedding(num_speakers + 1, 32)
        self.party_emb = nn.Embedding(num_parties + 1, 16)
        self.state_emb = nn.Embedding(num_states + 1, 16)
        self.subject_emb = nn.Embedding(num_subjects + 1, 16)
        
        self.history_layer = nn.Sequential(nn.Linear(5, 32), nn.ReLU())

        # Fusion
        fusion_size = 768 + 32 + 16 + 16 + 16 + 32 # 880
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.LayerNorm(256),  # <--- Changed from BatchNorm1d,
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, speaker, party, state, subject, history):
        # Text Path
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_vec = roberta_out.last_hidden_state[:, 0, :] 
        
        # Metadata Path
        spk_vec = self.speaker_emb(speaker)
        pty_vec = self.party_emb(party)
        stt_vec = self.state_emb(state)
        sbj_vec = self.subject_emb(subject)
        hist_vec = self.history_layer(history)
        
        # Fusion
        combined_vec = torch.cat((text_vec, spk_vec, pty_vec, stt_vec, sbj_vec, hist_vec), dim=1)
        
        # Classify
        logits = self.classifier(combined_vec)
        return logits