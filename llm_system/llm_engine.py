import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load env variables immediately
load_dotenv()

class LLMVerifier:
    def __init__(self):
        # 1. Initialize client as None first (Prevents AttributeError)
        self.client = None
        
        # 2. Get API Key (Try env var, or paste it here as fallback)
        self.api_key = os.getenv("GROQ_API_KEY") 
        # If env var didn't work, paste key here:
        # self.api_key = "gsk_..." 

        self.model_name = "llama-3.3-70b-versatile"
        
        if not self.api_key:
            print("❌ ERROR: API Key not found. Please set GROQ_API_KEY in .env or code.")
            return

        try:
            self.client = Groq(api_key=self.api_key)
            print(f"--- GROQ CLOUD ENGINE LOADED ({self.model_name}) ---")
        except Exception as e:
            print(f"❌ Error initializing Groq: {e}")

    def verify(self, statement, hcat_result=None):
        """
        Analyse EXPERTE avec le LLM Groq - Prompt optimisé
        """
        # Safety Check: If client failed to load, return error instead of crashing
        if self.client is None:
            return "ERROR", 0.0, "API Client not initialized (Check API Key)"
        
        try:
            # --- CONSTRUCT PROMPT ---
            if hcat_result:
                # Mode: Double Validation (Expert Context)
                prompt = f"""Tu es un fact-checker expert senior avec 15 ans d'expérience.

📰 ARTICLE À VÉRIFIER:
"{statement}"

🤖 PRÉ-ANALYSE HCAT (Intelligence Artificielle):
- Prédiction: {hcat_result['prediction']}
- Confiance: {hcat_result['confidence']*100:.1f}% ⚠️ FAIBLE - Nécessite validation experte
- Prob FAKE: {hcat_result['probabilities']['FAKE']*100:.1f}% | Prob REAL: {hcat_result['probabilities']['REAL']*100:.1f}%

🎯 TA MISSION:
Effectue une analyse RIGOUREUSE.

📋 GRILLE D'ANALYSE (Mental Scoring):
1. SOURCES ET CRÉDIBILITÉ (Source reconnue? Style journalistique?)
2. RED FLAGS (Affirmations impossibles? Complotisme? Langage sensationnaliste?)
3. COHÉRENCE FACTUELLE (Timeline logique? Chiffres réalistes?)
4. INDICES LINGUISTIQUES (Ton neutre ou militant?)
5. PLAUSIBILITÉ CONTEXTUELLE

📊 FORMAT DE RÉPONSE STRICT (JSON UNIQUEMENT):
{{
    "prediction": "FAKE" ou "REAL",
    "confidence": 0.70 à 1.0,
    "reasoning": "Analyse concise en 3-4 phrases",
    "key_red_flags": ["flag 1", "flag 2"],
    "scores": {{
        "sources": 0, "red_flags": 0, "factual": 0, "linguistic": 0, "plausibility": 0, "total": 0
    }},
    "expert_note": "Note finale"
}}"""
            else:
                # Mode: Short / Direct Fact Check
                prompt = f"""Tu es un fact-checker expert senior. Analyse cette affirmation.

💬 AFFIRMATION: "{statement}"

🔬 MÉTHODOLOGIE:
1. VÉRIFIABILITÉ (Consensus scientifique/factuel?)
2. RED FLAGS (Théorie du complot? Contredit faits établis?)
3. CONTEXTE (Simplification excessive?)

📊 RÉPONDS UNIQUEMENT AVEC CE JSON:
{{
    "prediction": "FAKE" ou "REAL",
    "confidence": 0.70 à 1.0,
    "reasoning": "Explication en 2-3 phrases",
    "key_points": ["point 1", "point 2"],
    "fact_check_basis": "Consensus scientifique" ou "Fait vérifiable" ou "Analyse contextuelle"
}}"""

            # --- CALL API ---
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Tu es un fact-checker expert. Réponds UNIQUEMENT avec du JSON valide. Pas de markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            llm_text = completion.choices[0].message.content.strip()
            
            # --- CLEANUP JSON ---
            if "```" in llm_text:
                llm_text = llm_text.split("```")[1]
                if llm_text.startswith("json"):
                    llm_text = llm_text[4:]
            llm_text = llm_text.strip()
            
            # --- PARSE ---
            result = json.loads(llm_text)
            
            # Extract values safely
            verdict = result.get('prediction', 'UNCERTAIN')
            confidence = float(result.get('confidence', 0.5)) * 100
            
            # Build Explanation
            reason = f"{result.get('reasoning', '')}\n"
            if 'scores' in result:
                reason += f"   [Score Expert: {result['scores'].get('total', 0)}/50]"
            
            return verdict, confidence, reason

        except json.JSONDecodeError:
            return "ERROR", 0.0, f"LLM Output format error: {llm_text[:50]}..."
        except Exception as e:
            return "ERROR", 0.0, str(e)