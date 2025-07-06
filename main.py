from datasets import load_dataset

# Load the Bias in Bios dataset
dataset = load_dataset("LabHC/bias_in_bios")

# Print dataset info
print(dataset)

# Show a sample bio
example = dataset["train"][0]
print("Bio:", example["bio"])
print("Gender:", example["gender"])
print("Title:", example["title"])
