import pandas as pd

def clean_text(text):
    return str(text).lower().strip()[:300]

def main():
    print("--- VERIFYING LEAKAGE FOR AI DATASET ---")
    
    # Load ISOT
    print("Loading ISOT...")
    isot_true = pd.read_csv("liar_isot/data/True.csv")
    isot_fake = pd.read_csv("liar_isot/data/Fake.csv")
    
    train_set = set()
    for text in isot_true['text']: train_set.add(clean_text(text))
    for text in isot_fake['text']: train_set.add(clean_text(text))
        
    # Load New File
    print("Loading AI Test File...")
    new_df = pd.read_csv("liar_isot/data/long_ai_test_100.csv")
    
    leak_count = 0
    for text in new_df['text']:
        if clean_text(text) in train_set:
            leak_count += 1
            
    print(f"\nOverlapping Samples: {leak_count}")
    if leak_count == 0:
        print("✅ SUCCESS: 0% Leakage. This test is valid.")
    else:
        print("❌ WARNING: Still leaking.")

if __name__ == "__main__":
    main()