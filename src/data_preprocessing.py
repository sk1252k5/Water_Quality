# importing necessary libraries
import pandas as pd
import numpy as np
import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    return df

def dataset_summary(df):
    print("\nColumn Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())

def clean_dataset(df):
    # Fill missing values using median (best for environmental data)
    df_clean = df.copy()
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    return df_clean


def apply_standards(df):
    df['pH_Status'] = df['ph'].apply(
        lambda x: 'Normal' if 6.5 <= x <= 8.5 else 'Abnormal'
    )

    df['TDS_Status'] = df['Solids'].apply(
        lambda x: 'Normal' if x <= 500 else 'High'
    )

    df['Turbidity_Status'] = df['Turbidity'].apply(
        lambda x: 'Normal' if x <= 5 else 'High'
    )

    return df


