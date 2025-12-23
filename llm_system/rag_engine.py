import wikipedia
from sentence_transformers import CrossEncoder
import torch
import numpy as np

class RagVerifier:
    def __init__(self):
        print("Loading RAG Fact-Checker (Cross-Encoder)...")
        # 0 = Contradiction, 1 = Entailment, 2 = Neutral
        self.model = CrossEncoder('cross-encoder/nli-distilroberta-base')

    def search_wiki(self, query):
        """Searches Wikipedia for relevant summaries"""
        print(f"  [RAG] Searching Wikipedia for: '{query[:50]}...'")
        evidence_list = []
        
        try:
            # 1. Search for Page Titles
            # We assume the most relevant nouns are at the start/end or the whole sentence works
            search_results = wikipedia.search(query, results=2)
            
            if not search_results:
                return []

            # 2. Get Summaries for the top pages
            for title in search_results:
                try:
                    # Get the first 3 sentences of the page summary
                    # auto_suggest=False prevents weird redirects
                    summary = wikipedia.summary(title, sentences=3, auto_suggest=False)
                    evidence_list.append(f"{title}: {summary}")
                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                    continue
                    
        except Exception as e:
            print(f"  [RAG Error] Wiki failed: {e}")
            
        return evidence_list

    def verify(self, claim):
        # 1. Search
        evidence_list = self.search_wiki(claim)
        
        if not evidence_list:
            return "UNKNOWN", 0.0, "No relevant Wikipedia pages found."

        # 2. Compare Claim vs Evidence
        pairs = [(claim, ev) for ev in evidence_list]
        scores = self.model.predict(pairs)
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()
        
        # 3. Logic
        max_fake = np.max(probs[:, 0]) # Contradiction score
        max_real = np.max(probs[:, 1]) # Entailment score
        
        print(f"  [RAG Scores] Real: {max_real:.2f} | Fake: {max_fake:.2f}")

        best_idx = -1
        
        # Thresholds: Wikipedia is factual. If it agrees, it's REAL.
        if max_real > max_fake and max_real > 0.2:
            best_idx = np.argmax(probs[:, 1])
            return "REAL", float(max_real) * 100, evidence_list[best_idx]
            
        elif max_fake > max_real and max_fake > 0.2:
            best_idx = np.argmax(probs[:, 0])
            return "FAKE", float(max_fake) * 100, evidence_list[best_idx]
            
        else:
            return "UNCERTAIN", 50.0, "Wikipedia did not confirm or deny this specifically."