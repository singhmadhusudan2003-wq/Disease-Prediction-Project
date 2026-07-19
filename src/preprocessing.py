# preprocessing.py
# Helper functions to clean and prepare the heart disease dataset for
# training (train.py) and inference (predict.py / app.py).

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns present in the raw UCI-style CSV that are NOT clinical features and
# must never be used to train the model:
#   - id      : row identifier, has no predictive meaning.
#   - num     : original multi-class label (0-4), replaced by the binary
#               "target" column derived below.
#   - dataset : name of the clinical site that collected the record
#               (Cleveland, Hungary, Switzerland, VA Long Beach). This is a
#               data-collection artifact, not a patient attribute. Keeping it
#               causes data leakage, since the model could learn "which
#               hospital" as a shortcut correlated with the label instead of
#               a real clinical pattern, and it can never be supplied for a
#               new patient at prediction time.
NON_FEATURE_COLUMNS = ["id", "num", "dataset"]


def load_data(path):
    """
    Load the raw heart disease CSV and convert it into a clean, model-ready
    schema.

    Steps performed:
        1. Read the CSV, treating '?' as a missing value marker.
        2. Derive a binary "target" column from the original "num" column.
        3. Drop non-clinical columns (see NON_FEATURE_COLUMNS).

    Column names are preserved exactly as they appear in the raw CSV (e.g.
    "thalch" is NOT renamed here). The raw dataset is treated as the single
    source of truth for naming; any naming adaptation needed for user-facing
    input (predict.py / app.py) is handled in that layer, not here.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only clinical feature columns plus "target".
    """
    try:
        df = pd.read_csv(path, na_values="?")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            "Run src/make_dataset.py or check the DATA_PATH used in train.py."
        ) from exc

    if "num" not in df.columns:
        raise KeyError(
            "Expected a 'num' column in the raw dataset to derive the binary "
            "target, but it was not found. Check that the correct raw CSV "
            "(UCI Cleveland/Kaggle heart disease format) was provided."
        )

    # Create binary target from the original UCI multi-class label:
    # num = 0 -> No Disease, num = 1-4 -> Disease
    df["target"] = (df["num"] > 0).astype(int)

    # Drop identifier, original label, and non-clinical site column.
    df = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")

    return df


def handle_missing_values(df):
    """
    Fill missing values column by column: median for numeric columns,
    mode for categorical columns. Returns a new DataFrame (does not mutate
    the input in place).
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isna().sum() > 0:

            # Numeric column
            if is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

            # Categorical column
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def split_features_target(df):
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def get_train_test_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split, preserving the target class balance."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_features(X_train, X_test):
    """Fit a StandardScaler on the training features and apply it to both sets."""
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


def clean_and_prepare(path, test_size=0.2):
    """
    Full preprocessing pipeline: load, clean, encode, split, and scale the
    heart disease dataset.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.
    test_size : float
        Fraction of the data to hold out for testing.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # Load data
    df = load_data(path)

    # Fill missing values
    df = handle_missing_values(df)

    # Convert categorical columns to numeric (one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)

    # Split X and y
    X, y = split_features_target(df)

    # Train/Test Split
    X_train, X_test, y_train, y_test = get_train_test_data(
        X,
        y,
        test_size=test_size
    )

    # Scale features
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X.columns
    )
