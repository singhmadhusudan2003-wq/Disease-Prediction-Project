# make_dataset.py
#
# LEGACY / REFERENCE ONLY - not used by the current training pipeline.
#
# Generates a synthetic heart-disease-like dataset for demo/testing purposes.
# It was originally used as a stand-in when the real dataset wasn't
# available. The project now trains on the real dataset at data/heart.csv,
# whose schema (string categorical values, a "num" label column, a
# multi-site "dataset" column) does not match this script's simplified
# numeric-coded output. Running this script does NOT affect data/heart.csv
# or the trained model - it writes to a separate file (see OUTPUT_PATH).

import sys

import numpy as np
import pandas as pd

OUTPUT_PATH = 'data/synthetic_heart_sample.csv'
N_SAMPLES = 1000
RANDOM_SEED = 42


def make_heart_data(n):
    """Generate `n` rows of synthetic heart-disease-like data with a
    binary target loosely correlated with the generated features."""
    age = np.random.randint(29, 78, n)
    sex = np.random.randint(0, 2, n)  # 1 = male, 0 = female
    cp = np.random.randint(0, 4, n)   # chest pain type 0-3
    trestbps = np.random.randint(94, 200, n)   # resting bp
    chol = np.random.randint(126, 565, n)      # cholesterol
    fbs = np.random.choice([0, 1], n, p=[0.85, 0.15])  # fasting blood sugar
    restecg = np.random.randint(0, 3, n)
    thalach = np.random.randint(71, 202, n)    # max heart rate
    exang = np.random.choice([0, 1], n, p=[0.68, 0.32])  # exercise induced angina
    oldpeak = np.round(np.random.uniform(0, 6.2, n), 1)
    slope = np.random.randint(0, 3, n)
    ca = np.random.randint(0, 4, n)     # number of major vessels
    thal = np.random.randint(0, 3, n)

    # Target loosely depends on the generated features: older age, high
    # chol, low thalach, exang=1, high oldpeak -> more risk.
    risk_score = (
        0.03 * age +
        0.15 * sex +
        0.35 * cp +
        0.01 * trestbps +
        0.005 * chol -
        0.02 * thalach +
        0.9 * exang +
        0.4 * oldpeak +
        0.3 * ca +
        0.25 * thal
    )

    # Add noise so it's not trivially separable.
    noise = np.random.normal(0, 2.5, n)
    risk_score = risk_score + noise

    # Convert risk score to a 0/1 target using a median threshold.
    threshold = np.median(risk_score)
    target = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })

    return df


def main():
    np.random.seed(RANDOM_SEED)

    df = make_heart_data(N_SAMPLES)

    # Add a few missing values randomly, since real-world data is never clean.
    missing_idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[missing_idx, 'chol'] = np.nan

    missing_idx2 = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_idx2, 'thalach'] = np.nan

    try:
        df.to_csv(OUTPUT_PATH, index=False)
    except OSError as exc:
        print(f"Error: could not write synthetic dataset to '{OUTPUT_PATH}'. {exc}")
        sys.exit(1)

    print(f"Synthetic dataset created at '{OUTPUT_PATH}', shape: {df.shape}")
    print(df['target'].value_counts())


if __name__ == '__main__':
    main()
