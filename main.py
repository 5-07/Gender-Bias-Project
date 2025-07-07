# main.py
from data_utils import load_pairs
from model_utils import load_model, predict_in_batches
from analysis_utils import compute_metrics, save_results
import plots

MODEL_PATH = "./trained_model"
CSV_PATH = "bios_pairs.csv"
OUTPUT_CSV = "counterfactual_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df_pairs = load_pairs(CSV_PATH, n_samples=1000)

# Load model
model, tokenizer = load_model(MODEL_PATH, DEVICE)

# Predict
preds_orig, conf_orig = predict_in_batches(
    df_pairs["original_text"].tolist(),
    model,
    tokenizer,
    batch_size=16,
    device=DEVICE
)

preds_cf, conf_cf = predict_in_batches(
    df_pairs["swapped_text"].tolist(),
    model,
    tokenizer,
    batch_size=16,
    device=DEVICE
)

# Metrics
flipped, flip_rate, conf_diff, avg_shift = compute_metrics(
    preds_orig, preds_cf, conf_orig, conf_cf
)

print(f"Flip Rate: {flip_rate:.4f}")
print(f"Average Confidence Shift: {avg_shift:.4f}")

# Save results
save_results(
    df_pairs,
    preds_orig,
    conf_orig,
    preds_cf,
    conf_cf,
    flipped,
    conf_diff,
    OUTPUT_CSV
)

# Run plots
plots.plot_all(OUTPUT_CSV)
