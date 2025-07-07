import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import numpy as np
from tqdm import tqdm

def load_model(model_path, device):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_in_batches(texts, model, tokenizer, batch_size, device):
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

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits

            preds = logits.argmax(dim=-1).cpu().numpy()
            confs = logits.softmax(dim=-1).max(dim=-1).values.cpu().numpy()

            all_preds.extend(preds)
            all_confs.extend(confs)

    return np.array(all_preds), np.array(all_confs)
