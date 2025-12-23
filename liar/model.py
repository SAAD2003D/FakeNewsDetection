import torch
import torch.nn as nn
from transformers import AutoModel

class HCATModel(nn.Module):
    def __init__(self, num_speakers, num_parties, num_states, num_subjects):
        super(HCATModel, self).__init__()
        
        # --- Branch A: Text Processing (RoBERTa) ---
        print("Loading RoBERTa model... (This might take a moment)")
        self.roberta = AutoModel.from_pretrained('roberta-base')
        
        # Optimization: Freeze the early layers of RoBERTa
        # We only want to train the top layers to save GPU memory and prevent overfitting.
        # 1. Freeze the Embeddings (Word -> Vector)
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
            
        # 2. Freeze the first 8 layers (out of 12)
        # We only let the model update the last 4 layers.
       # for layer in self.roberta.encoder.layer[:8]:
         #   for param in layer.parameters():
         #       param.requires_grad = False
        
        # --- Branch B: Metadata Processing (Embeddings) ---
        # We map IDs to dense vectors.
        # (Input Dictionary Size, Output Vector Size)
        self.speaker_emb = nn.Embedding(num_speakers + 1, 32) # +1 for unknown
        self.party_emb = nn.Embedding(num_parties + 1, 16)
        self.state_emb = nn.Embedding(num_states + 1, 16)
        self.subject_emb = nn.Embedding(num_subjects + 1, 16)
        
        # Processing the Credit History (5 numbers)
        self.history_layer = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )

        # --- Branch C: Fusion & Classification ---
        # Calculate total input size for the classifier:
        # 768 (RoBERTa [CLS]) + 32 (Speaker) + 16 (Party) + 16 (State) + 16 (Subject) + 32 (History)
        fusion_size = 768 + 32 + 16 + 16 + 16 + 32  # Total = 880
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(fusion_size, 256),
            nn.BatchNorm1d(256), # Normalize data to train faster
            nn.ReLU(),
            nn.Dropout(0.4),     # Drop 40% of connections to prevent memorization
            
            # Layer 2
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer (6 Classes for LIAR labels)
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, speaker, party, state, subject, history):
        # 1. Text Path
        # Pass text to RoBERTa
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the [CLS] token (the first token of the sequence) which represents the whole sentence
        text_vec = roberta_out.last_hidden_state[:, 0, :] 
        
        # 2. Metadata Path
        spk_vec = self.speaker_emb(speaker)
        pty_vec = self.party_emb(party)
        stt_vec = self.state_emb(state)
        sbj_vec = self.subject_emb(subject)
        
        hist_vec = self.history_layer(history)
        
        # 3. Fusion Path (Concatenation)
        # Glue all vectors together side-by-side
        combined_vec = torch.cat((text_vec, spk_vec, pty_vec, stt_vec, sbj_vec, hist_vec), dim=1)
        
        # 4. Classification
        logits = self.classifier(combined_vec)
        
        return logits