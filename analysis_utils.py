import numpy as np
import pandas as pd

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

stem_professions = [
    6, 8, 9, 18, 19, 21, 23, 24
]

def compute_metrics(preds_orig, preds_cf, conf_orig, conf_cf):
    flipped = preds_orig != preds_cf
    flip_rate = flipped.sum() / len(flipped)
    conf_diff = np.abs(conf_orig - conf_cf)
    avg_shift = np.mean(conf_diff)
    return flipped, flip_rate, conf_diff, avg_shift

def save_results(df_pairs, preds_orig, conf_orig, preds_cf, conf_cf, flipped, conf_diff, output_csv):
    df_pairs["original_pred"] = preds_orig
    df_pairs["original_conf"] = conf_orig
    df_pairs["counterfactual_pred"] = preds_cf
    df_pairs["counterfactual_conf"] = conf_cf
    df_pairs["flipped"] = flipped
    df_pairs["confidence_shift"] = conf_diff

    # Map numeric labels to text
    df_pairs["original_pred_label"] = df_pairs["original_pred"].map(profession_map)
    df_pairs["counterfactual_pred_label"] = df_pairs["counterfactual_pred"].map(profession_map)
    df_pairs["profession_label"] = df_pairs["profession"].map(profession_map)

    # Add STEM vs non-STEM
    df_pairs["STEM_Category"] = df_pairs["profession"].apply(
        lambda x: "STEM" if x in stem_professions else "Non-STEM"
    )

    df_pairs.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
