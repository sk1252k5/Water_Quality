import pandas as pd
from src.data_preprocessing import load_dataset, clean_dataset, apply_standards

def recommend_treatment(row):
    treatments = []

    # pH correction
    if row['ph'] < 6.5 or row['ph'] > 8.5:
        treatments.append("Boiling / pH Adjustment")

    # High TDS
    if row['Solids'] > 500:
        treatments.append("RO Filtration")

    # High turbidity
    if row['Turbidity'] > 5:
        treatments.append("Sediment Filter")

    # Unsafe water
    if row['Potability'] == 0:
        treatments.append("UV Purification")

    if not treatments:
        return "No Treatment Required"

    return " + ".join(sorted(set(treatments)))


def main():
    # Load raw data
    df = load_dataset("data/raw/water_potability.csv")

    # Clean data
    df = clean_dataset(df)

    # Apply standards
    df = apply_standards(df)

    # Generate recommendations
    df['Recommended_Treatment'] = df.apply(recommend_treatment, axis=1)

    # Save final dataset
    df.to_csv("data/processed/water_treatment_clean.csv", index=False)
    print("✅ Clean dataset with purification recommendations saved.")


if __name__ == "__main__":
    main()
