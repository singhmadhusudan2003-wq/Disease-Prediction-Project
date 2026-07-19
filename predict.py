# predict.py
# Command-line interface for predicting heart disease from patient details.
# Run with: python predict.py

import sys

import joblib
import pandas as pd

MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

# ---------------------------------------------------------------------------
# Optional cosmetic labels and sanity-check ranges for known raw columns.
#
# IMPORTANT: these dicts only affect prompt wording and numeric validation
# bounds. They do NOT determine which fields exist, which are categorical,
# or what dummy columns get built - that is always derived dynamically from
# feature_names.pkl at runtime (see build_field_schema()). A column that
# isn't listed here still works correctly, just with a generic label and a
# broad default numeric range. This is what lets predict.py keep working,
# without code changes, if train.py is rerun with a different feature set.
# ---------------------------------------------------------------------------
FIELD_LABELS = {
    'age': 'Age (years)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'thalch': 'Maximum heart rate achieved',
    'oldpeak': 'ST depression induced by exercise (oldpeak)',
    'ca': 'Number of major vessels colored by fluoroscopy',
    'sex': 'Sex',
    'cp': 'Chest pain type',
    'fbs': 'Fasting blood sugar > 120 mg/dl',
    'restecg': 'Resting ECG results',
    'exang': 'Exercise induced angina',
    'slope': 'Slope of the peak exercise ST segment',
    'thal': 'Thalassemia',
}

NUMERIC_RANGES = {
    'age': (1, 120),
    'trestbps': (50, 250),
    'chol': (50, 700),
    'thalch': (50, 250),
    'oldpeak': (-5.0, 10.0),
    'ca': (0, 3),
}

DEFAULT_NUMERIC_RANGE = (-1000.0, 1000.0)

# ---------------------------------------------------------------------------
# Cosmetic-only label for the baseline category of each categorical field -
# the category pandas.get_dummies(drop_first=True) drops during training,
# which therefore never appears in feature_names.pkl. This dict has NO
# effect on encoding: it exists purely so the CLI menu can show a
# meaningful name instead of a generic placeholder. A field missing from
# this dict (e.g. a brand-new categorical column after a future retrain)
# still works correctly via DEFAULT_BASELINE_LABEL.
#
# Values were verified against the current training data (data/heart.csv):
# pandas.get_dummies(drop_first=True) drops whichever category is first in
# alphabetical order, e.g. restecg's alphabetical order is
# "lv hypertrophy" < "normal" < "st-t abnormality", so "lv hypertrophy" is
# the dropped baseline - not "normal".
BASELINE_LABELS = {
    'sex': 'Female',
    'cp': 'Asymptomatic',
    'fbs': 'False',
    'restecg': 'Lv Hypertrophy',
    'exang': 'False',
    'slope': 'Downsloping',
    'thal': 'Fixed Defect',
}

DEFAULT_BASELINE_LABEL = 'Other / none of the listed categories'


def load_artifacts():
    """
    Load the trained model, scaler, and feature name list saved by
    train.py.

    Exits with a clear message (instead of an unhandled traceback) if any
    artifact file is missing.
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    except FileNotFoundError as exc:
        print(f"Error: could not load model artifacts ({exc}).")
        print("Please run 'python train.py' first to train and save the model.")
        sys.exit(1)

    return model, scaler, feature_names


def build_field_schema(feature_names):
    """
    Derive the set of raw input fields - and, for categorical fields, their
    known non-baseline category values - directly from `feature_names`.

    Any feature name containing an underscore is treated as a one-hot dummy
    column produced by pandas.get_dummies, in the form
    "<raw_column>_<category_value>" (e.g. "cp_typical angina" comes from raw
    column "cp" with category value "typical angina"). Feature names without
    an underscore are treated as numeric columns.

    No column names or category values are hardcoded: the fields prompted
    for, and the valid choices offered, always match whatever the currently
    loaded model was actually trained on. If train.py is rerun with a
    different feature set, predict.py adapts automatically.

    Parameters
    ----------
    feature_names : list of str
        Trained feature columns, as saved in models/feature_names.pkl.

    Returns
    -------
    tuple
        (numeric_columns, categorical_fields)
        numeric_columns : list of str
        categorical_fields : dict mapping raw_column -> list of
            (dummy_column, category_value) pairs, in the order they appear
            in feature_names.
    """
    numeric_columns = []
    categorical_fields = {}

    for column in feature_names:
        if '_' in column:
            raw_column, value = column.split('_', 1)
            categorical_fields.setdefault(raw_column, []).append(
                (column, value)
            )
        else:
            numeric_columns.append(column)

    return numeric_columns, categorical_fields


def prompt_float(label, value_range):
    """Prompt for a number within value_range, retrying on bad input."""
    min_value, max_value = value_range

    while True:
        raw = input(f"{label} [{min_value}-{max_value}]: ").strip()

        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a valid number.")
            continue

        if not (min_value <= value <= max_value):
            print(f"  Value must be between {min_value} and {max_value}.")
            continue

        return value


def prompt_choice(label, dummy_options, baseline_label):
    """
    Prompt the user to choose a category for one categorical field.

    `dummy_options` is the list of (dummy_column, category_value) pairs for
    this field, taken from feature_names.pkl - this is the only thing that
    determines which real dummy columns exist and can be selected.

    `baseline_label` is used ONLY to display a meaningful menu entry for
    the baseline category (the one training dropped, which never appears
    in feature_names.pkl). It is a UI label, nothing more: selecting it
    always returns None, which get_patient_input() uses to leave every
    dummy column for this field at 0 - the same encoding baseline
    categories always had. Changing or removing a baseline_label can never
    change what gets encoded, only what the menu displays.

    Returns
    -------
    str or None
        The dummy column name to set to 1, or None if the baseline
        category was selected.
    """
    print(f"{label}:")
    for i, (_dummy_column, value) in enumerate(dummy_options, start=1):
        print(f"  {i}. {value}")

    baseline_choice = len(dummy_options) + 1
    print(f"  {baseline_choice}. {baseline_label}")

    while True:
        raw = input(f"Enter choice [1-{baseline_choice}]: ").strip()

        if not raw.isdigit():
            print("  Please enter a number from the list.")
            continue

        choice = int(raw)
        if not (1 <= choice <= baseline_choice):
            print(f"  Please enter a number between 1 and {baseline_choice}.")
            continue

        if choice == baseline_choice:
            return None

        return dummy_options[choice - 1][0]


def get_patient_input(feature_names):
    """
    Collect and validate patient details from the terminal, using a field
    schema derived dynamically from feature_names.pkl (see
    build_field_schema).

    Returns
    -------
    dict
        A pre-encoded feature vector matching `feature_names`: each numeric
        column holds the entered value, and at most one dummy column per
        categorical field is set to 1 (none, if the baseline category was
        chosen). Every other column is 0.
    """
    numeric_columns, categorical_fields = build_field_schema(feature_names)

    encoded = {column: 0.0 for column in feature_names}

    print("\nPlease enter patient details:\n")

    for column in numeric_columns:
        label = FIELD_LABELS.get(column, column)
        value_range = NUMERIC_RANGES.get(column, DEFAULT_NUMERIC_RANGE)
        encoded[column] = prompt_float(label, value_range)

    for raw_column, dummy_options in categorical_fields.items():
        label = FIELD_LABELS.get(raw_column, raw_column)
        baseline_label = BASELINE_LABELS.get(raw_column, DEFAULT_BASELINE_LABEL)
        selected_dummy = prompt_choice(label, dummy_options, baseline_label)
        if selected_dummy is not None:
            encoded[selected_dummy] = 1

    return encoded


def encode_patient_data(encoded_dict, feature_names):
    """
    Turn an already-encoded feature dict (see get_patient_input) into the
    single-row DataFrame the scaler/model expect, in the exact column
    order feature_names specifies.
    """
    return pd.DataFrame([encoded_dict], columns=feature_names)


def predict_from_data(model, scaler, feature_names, encoded_dict):
    """Run the full inference pipeline on a single pre-encoded patient record."""
    patient_df = encode_patient_data(encoded_dict, feature_names)
    patient_scaled = scaler.transform(patient_df)

    prediction = model.predict(patient_scaled)[0]

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(patient_scaled)[0][1]
    else:
        probability = None

    return prediction, probability


def main():
    print("Loading model...")
    model, scaler, feature_names = load_artifacts()

    try:
        user_data = get_patient_input(feature_names)
    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled. Exiting.")
        sys.exit(0)

    try:
        prediction, probability = predict_from_data(
            model, scaler, feature_names, user_data
        )
    except Exception as exc:
        print(f"Error: could not generate a prediction. {exc}")
        sys.exit(1)

    print("\n----- Prediction Result -----")

    if prediction == 1:
        print("Result: Heart Disease Detected")
    else:
        print("Result: No Heart Disease")

    if probability is not None:
        print(f"Probability: {probability * 100:.2f}%")

    print("\nNote: This is a student ML project, not medical advice.")


if __name__ == '__main__':
    main()
