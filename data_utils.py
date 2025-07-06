from datasets import load_dataset
import pandas as pd
import re

# Gender swap dictionary
swap_dict = {
    "he": "she",
    "she": "he",
    "his": "her",
    "her": "his",
    "him": "her",
    "man": "woman",
    "woman": "man",
    "father": "mother",
    "mother": "father",
    "boy": "girl",
    "girl": "boy",
    "male": "female",
    "female": "male",
    "himself": "herself",
    "herself": "himself"
}

def explore_data():
    dataset = load_dataset("LabHC/bias_in_bios")
    df = dataset["train"].to_pandas()

    # Print sample rows
    print(df.sample(5))
    print("Number of professions:", df["profession"].nunique())
    print("Gender balance:")
    print(df["gender"].value_counts())

def swap_gender_terms(text, swap_dict):
    pattern = re.compile(r'\b(' + '|'.join(swap_dict.keys()) + r')\b', re.IGNORECASE)
    
    def replacer(match):
        word = match.group(0)
        swapped = swap_dict.get(word.lower(), word)
        return swapped.capitalize() if word[0].isupper() else swapped
    
    return pattern.sub(replacer, text)

def create_counterfactual_pairs():
    dataset = load_dataset("LabHC/bias_in_bios")
    df = dataset["train"].to_pandas()

    pairs = []

    for idx, row in df.iterrows():
        original = row["hard_text"]
        swapped = swap_gender_terms(original, swap_dict)
        pairs.append({
            "original_text": original,
            "swapped_text": swapped,
            "profession": row["profession"],
            "gender": row["gender"]
        })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv("bios_pairs.csv", index=False)
    print("Saved bios_pairs.csv with counterfactual pairs!")
