import pandas as pd

df_pairs = pd.read_csv("bios_pairs.csv")
print(df_pairs.head())
#To test on og bios
orig_encodings = tokenizer(
    df_pairs["original_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

model.eval()
with torch.no_grad():
    outputs_orig = model(**orig_encodings)
    preds_orig = outputs_orig.logits.argmax(dim=-1).numpy()
    conf_orig = outputs_orig.logits.softmax(dim=-1).max(dim=-1).values.numpy()

#To test on cf bios
cf_encodings = tokenizer(
    df_pairs["counterfactual_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

with torch.no_grad():
    outputs_cf = model(**cf_encodings)
    preds_cf = outputs_cf.logits.argmax(dim=-1).numpy()
    conf_cf = outputs_cf.logits.softmax(dim=-1).max(dim=-1).values.numpy()
