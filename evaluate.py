from sklearn.metrics import classification_report

# Load test data
test_df = dataset["test"].to_pandas()

# Tokenize test bios
test_encodings = tokenizer(
    test_df["hard_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt",
)

# Run model on test set
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    predictions = outputs.logits.argmax(dim=-1).numpy()

# Evaluate
print(classification_report(
    test_df["profession"], predictions
))
