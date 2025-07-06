from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("LabHC/bias_in_bios")

# Convert train split to pandas
df = dataset["train"].to_pandas()

# Show some rows
print(df.sample(5))
