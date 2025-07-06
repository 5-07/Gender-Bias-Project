from data_utils import explore_data, create_counterfactual_pairs
from model_utils import load_bios_pairs, create_label_mapping, train_model

# Step 1: Explore
explore_data()

# Step 2: Create counterfactuals
create_counterfactual_pairs()

# Step 3: Load CSV
df = load_bios_pairs()

# Sample fewer rows coz my lappy crashed
df = df.sample(1000, random_state=42)

# Step 4: Map labels
df, prof2id, id2prof = create_label_mapping(df)

# Step 5: Train model
model, tokenizer = train_model(df)
