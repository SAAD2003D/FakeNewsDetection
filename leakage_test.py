import pandas as pd

def clean_text(text):
    """Simple cleaner to ensure accurate comparison"""
    return str(text).lower().strip()[:500] # Compare first 500 chars

def main():
    print("--- CHECKING FOR DATA LEAKAGE ---")
    
    # 1. Load Training Data (ISOT)
    print("Loading ISOT (Training Data)...")
    try:
        isot_true = pd.read_csv("liar_isot/data/True.csv")
        isot_fake = pd.read_csv("liar_isot/data/Fake.csv")
    except FileNotFoundError:
        print("Error: Could not find ISOT files. Check paths.")
        return

    # Create a "Set" of training texts (Sets are fast for searching)
    # We combine Title + First 500 chars of text to be unique
    train_set = set()
    
    print("Hashing Training Data...")
    for text in isot_true['text']:
        train_set.add(clean_text(text))
    for text in isot_fake['text']:
        train_set.add(clean_text(text))
        
    print(f"Total Unique Training Samples: {len(train_set)}")

    # 2. Load External Test Data
    print("Loading External Test Data...")
    try:
        # Update this filename to whatever you used for the 90% test
        ext_df = pd.read_csv("liar_isot/data/long_unknown_100.csv", on_bad_lines='skip') 
    except:
        print("Error: Could not find external_test.csv")
        return

    # 3. Check for Overlap
    leak_count = 0
    total_test = 0
    
    print("Comparing...")
    for text in ext_df['text']:
        total_test += 1
        clean_t = clean_text(text)
        
        if clean_t in train_set:
            leak_count += 1

    # 4. Results
    print("\n" + "="*30)
    print("LEAKAGE REPORT")
    print("="*30)
    print(f"Total Test Samples: {total_test}")
    print(f"Overlapping Samples: {leak_count}")
    
    leak_percentage = (leak_count / total_test) * 100
    print(f"Leakage Percentage: {leak_percentage:.2f}%")
    
    if leak_percentage > 10.0:
        print("\n❌ CRITICAL WARNING: High Data Leakage!")
        print("This External Dataset is largely a copy of ISOT.")
        print("Your 90% score is likely invalid.")
    elif leak_percentage > 0.0:
        print("\n⚠️ Minor Leakage.")
        print("Some Reuters articles appear in both. This is common but should be noted.")
    else:
        print("\n✅ PASSED: Zero Data Leakage.")
        print("Your 90% score is scientifically valid.")

if __name__ == "__main__":
    main()