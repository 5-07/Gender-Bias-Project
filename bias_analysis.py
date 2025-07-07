import pandas as pd
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
# ----------------------------------------

MODEL_PATH = "./trained_model"
CSV_PATH = "bios_pairs.csv"
OUTPUT_CSV = "counterfactual_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("Running on device:", DEVICE)

# ----------------------------------------
# PROFESSION MAPPING
# ----------------------------------------

# Mapping numeric labels to profession names
profession_map = {
    0: "accountant",
    1: "actor",
    2: "architect",
    3: "artist",
    4: "attorney",
    5: "chiropractor",
    6: "dentist",
    7: "dietitian",
    8: "engineer",
    9: "financial analyst",
    10: "interior designer",
    11: "journalist",
    12: "lawyer",
    13: "mechanic",
    14: "nurse",
    15: "painter",
    16: "personal trainer",
    17: "photographer",
    18: "physician",
    19: "professor",
    20: "psychologist",
    21: "researcher",
    22: "social worker",
    23: "software developer",
    24: "surgeon",
    25: "teacher",
    26: "therapist",
    27: "writer",
}

# Define STEM professions by their numeric labels
stem_professions = [
    6, 8, 9, 18, 19, 21, 23, 24
]

# ----------------------------------------
# LOAD TRAINED MODEL
# ----------------------------------------

print("Loading trained model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded!")

# ----------------------------------------
# LOAD COUNTERFACTUAL DATA
# ----------------------------------------

print("Loading bios_pairs.csv...")
df_pairs = pd.read_csv(CSV_PATH)

# Optional: work on a smaller sample for testing
df_pairs = df_pairs.sample(1000, random_state=42)

print("Number of rows loaded:", len(df_pairs))
print(df_pairs.head())

# ----------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------

def predict_in_batches(texts, model, tokenizer, batch_size=BATCH_SIZE):
    all_preds = []
    all_confs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing"):
        batch_texts = texts[i:i+batch_size]

        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits

            preds = logits.argmax(dim=-1).cpu().numpy()
            confs = logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()

            all_preds.extend(preds)
            all_confs.extend(confs)

    return np.array(all_preds), np.array(all_confs)

# ----------------------------------------
# RUN PREDICTIONS
# ----------------------------------------

# Original bios
print("\nPredicting on original bios...")
preds_orig, conf_orig = predict_in_batches(
    df_pairs["original_text"].tolist(),
    model,
    tokenizer
)

# Counterfactual bios
print("\nPredicting on counterfactual bios...")
preds_cf, conf_cf = predict_in_batches(
    df_pairs["swapped_text"].tolist(),
    model,
    tokenizer
)

# ----------------------------------------
# CALCULATE METRICS
# ----------------------------------------

print("\nCalculating metrics...")

flipped = preds_orig != preds_cf
flip_rate = flipped.sum() / len(flipped)
print(f"Flip Rate: {flip_rate:.4f}")

conf_diff = np.abs(conf_orig - conf_cf)
avg_shift = np.mean(conf_diff)
print(f"Average Confidence Shift: {avg_shift:.4f}")

# ----------------------------------------
# SAVE RESULTS
# ----------------------------------------

df_pairs["original_pred"] = preds_orig
df_pairs["original_conf"] = conf_orig
df_pairs["counterfactual_pred"] = preds_cf
df_pairs["counterfactual_conf"] = conf_cf
df_pairs["flipped"] = flipped
df_pairs["confidence_shift"] = conf_diff

# Map numeric labels to text for predictions
df_pairs["original_pred_label"] = df_pairs["original_pred"].map(profession_map)
df_pairs["counterfactual_pred_label"] = df_pairs["counterfactual_pred"].map(profession_map)
df_pairs["profession_label"] = df_pairs["profession"].map(profession_map)

# Add STEM/Non-STEM label
df_pairs["STEM_Category"] = df_pairs["profession"].apply(
    lambda x: "STEM" if x in stem_professions else "Non-STEM"
)

df_pairs.to_csv(OUTPUT_CSV, index=False)
print(f"Saved results to {OUTPUT_CSV}")

# ----------------------------------------
# VISUALIZATIONS
# ----------------------------------------

sns.set_theme(style="whitegrid")

results_df = pd.read_csv(OUTPUT_CSV)

# Flip rates by profession
flip_by_prof = results_df.groupby("profession_label")["flipped"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(
    x="profession_label",
    y="flipped",
    data=flip_by_prof,
    palette="viridis"
)
plt.xticks(rotation=45, ha="right")
plt.title("Flip Rate by Profession", fontsize=16)
plt.ylabel("Flip Rate")
plt.tight_layout()
plt.savefig("flip_rate_by_profession.png", dpi=300)
plt.show()

# Histogram of confidence shifts
plt.figure(figsize=(10,5))
sns.histplot(
    results_df["confidence_shift"],
    bins=30,
    kde=True,
    color="dodgerblue"
)
plt.title("Distribution of Confidence Shifts After Gender Swap", fontsize=16)
plt.xlabel("Absolute Confidence Change")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("confidence_shift_histogram.png", dpi=300)
plt.show()

# Violin plot for STEM vs Non-STEM
plt.figure(figsize=(8,5))
sns.violinplot(
    x="STEM_Category",
    y="confidence_shift",
    data=results_df,
    palette="Set2"
)
plt.title("Confidence Shifts by STEM Category", fontsize=16)
plt.tight_layout()
plt.savefig("confidence_violin_by_category.png", dpi=300)
plt.show()

# Heatmap of flip rates by category
flip_by_cat = results_df.groupby(["profession_label", "STEM_Category"])["flipped"].mean().reset_index()

heatmap_data = flip_by_cat.pivot(
    index="profession_label",
    columns="STEM_Category",
    values="flipped"
).fillna(0)

plt.figure(figsize=(10,12))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="rocket_r",
    linewidths=0.5
)
plt.title("Flip Rates Across Professions and STEM Categories", fontsize=16)
plt.tight_layout()
plt.savefig("flip_rate_heatmap.png", dpi=300)
plt.show()
