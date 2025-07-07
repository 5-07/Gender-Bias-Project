import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_all(csv_path):
    sns.set_theme(style="whitegrid")

    results_df = pd.read_csv(csv_path)

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

    # Heatmap of flip rates
    flip_by_cat = results_df.groupby(
        ["profession_label", "STEM_Category"]
    )["flipped"].mean().reset_index()

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
