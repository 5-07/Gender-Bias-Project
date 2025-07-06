import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the CSV
def load_bios_pairs(path="bios_pairs.csv"):
    df = pd.read_csv(path)
    return df

# Encode profession labels into numbers
def create_label_mapping(df):
    professions = sorted(df["profession"].unique())
    prof2id = {prof: idx for idx, prof in enumerate(professions)}
    id2prof = {idx: prof for prof, idx in prof2id.items()}
    df["label"] = df["profession"].map(prof2id)
    return df, prof2id, id2prof

# PyTorch Dataset class
class BiosDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Train the model
def train_model(df):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    texts = df["original_text"].tolist()
    labels = df["label"].tolist()

    dataset = BiosDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(df["label"].unique())
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

    print("Model training complete and saved!")

    return model, tokenizer
