# app.py
# Streamlit web app for heart disease prediction.
# Run using:
# streamlit run app.py

import streamlit as st

import predict

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# ---------------------------------------------------------------------------
# App-only cosmetic helpers.
#
# These do NOT affect encoding or predictions - they only control default
# starting values, number-input step size, and which visual column a field
# is placed in. The fields that actually get shown, their valid categories,
# and how they are encoded all come from predict.build_field_schema(),
# driven entirely by feature_names.pkl (same as the CLI). A field missing
# from these dicts still renders correctly with a sensible generic default.
# ---------------------------------------------------------------------------
DEFAULT_VALUES = {
    'age': 45,
    'trestbps': 120,
    'chol': 200,
    'thalch': 150,
    'oldpeak': 1.0,
    'ca': 0,
}

STEP_SIZES = {
    'oldpeak': 0.1,
}

# Preferred left-column fields, kept for visual continuity with the
# original layout. Any field not listed here (e.g. a new column after a
# future retrain) is simply rendered in the right column.
PREFERRED_LEFT_COLUMN = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg'}


@st.cache_resource(show_spinner="Loading model...")
def get_cached_artifacts():
    """
    Load the trained model, scaler, and feature names by reusing
    predict.load_artifacts() - so the CLI and the web app always load the
    exact same artifacts the exact same way.

    predict.load_artifacts() calls sys.exit(1) if a file is missing, which
    is correct behavior for a CLI but would otherwise crash the whole
    Streamlit server. That SystemExit is caught here so app.py can show a
    proper in-app error instead (see main()).
    """
    try:
        return predict.load_artifacts()
    except SystemExit:
        return None, None, None


def render_numeric_inputs(numeric_columns):
    """Render a number_input for each numeric field, returning a dict of
    column -> entered value."""
    values = {}

    for column in numeric_columns:
        label = predict.FIELD_LABELS.get(column, column)
        min_value, max_value = predict.NUMERIC_RANGES.get(
            column, predict.DEFAULT_NUMERIC_RANGE
        )
        default_value = DEFAULT_VALUES.get(column, (min_value + max_value) / 2)
        step = STEP_SIZES.get(column, 1.0)

        values[column] = st.number_input(
            label,
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(default_value),
            step=step,
        )

    return values


def render_categorical_inputs(categorical_fields):
    """Render a selectbox for each categorical field, returning a dict of
    raw_column -> chosen dummy column name (or None for the baseline
    category)."""
    values = {}

    for raw_column, dummy_options in categorical_fields.items():
        label = predict.FIELD_LABELS.get(raw_column, raw_column)
        baseline_label = predict.BASELINE_LABELS.get(
            raw_column, predict.DEFAULT_BASELINE_LABEL
        )

        # (display_label, dummy_column_or_None) pairs, so the widget shows
        # friendly text while the real dummy column name (or None, for the
        # baseline) is carried alongside it.
        options = [(value, dummy_column) for dummy_column, value in dummy_options]
        options.append((baseline_label, None))

        choice = st.selectbox(label, options, format_func=lambda opt: opt[0])
        values[raw_column] = choice[1]

    return values


def build_encoded_input(feature_names, numeric_values, categorical_choices):
    """
    Combine widget values into the pre-encoded feature dict expected by
    predict.predict_from_data() - the same shape predict.py's CLI builds
    in get_patient_input(). This is the only "assembly" step that differs
    between the CLI and the web app (reading input() vs reading widget
    state); the actual encoding/scaling/prediction is shared code.
    """
    encoded = {column: 0.0 for column in feature_names}
    encoded.update(numeric_values)

    for selected_dummy in categorical_choices.values():
        if selected_dummy is not None:
            encoded[selected_dummy] = 1

    return encoded


def main():
    st.title("❤️ Heart Disease Prediction System")
    st.write("Enter patient details below and click Predict.")

    st.sidebar.title("About")
    st.sidebar.info("""
This project predicts whether a patient is likely to have heart disease.

Models Used:
- Logistic Regression
- SVM
- Random Forest

Best model selected automatically.
""")

    model, scaler, feature_names = get_cached_artifacts()

    if model is None:
        st.error(
            "Could not load model artifacts. Please run `python train.py` "
            "first to train and save the model, then restart the app."
        )
        st.stop()

    numeric_columns, categorical_fields = predict.build_field_schema(feature_names)

    left_numeric = [c for c in numeric_columns if c in PREFERRED_LEFT_COLUMN]
    right_numeric = [c for c in numeric_columns if c not in PREFERRED_LEFT_COLUMN]
    left_categorical = {
        k: v for k, v in categorical_fields.items() if k in PREFERRED_LEFT_COLUMN
    }
    right_categorical = {
        k: v for k, v in categorical_fields.items() if k not in PREFERRED_LEFT_COLUMN
    }

    col1, col2 = st.columns(2)

    with col1:
        numeric_values = render_numeric_inputs(left_numeric)
        categorical_choices = render_categorical_inputs(left_categorical)

    with col2:
        numeric_values.update(render_numeric_inputs(right_numeric))
        categorical_choices.update(render_categorical_inputs(right_categorical))

    if st.button("Predict"):
        encoded_input = build_encoded_input(
            feature_names, numeric_values, categorical_choices
        )

        try:
            prediction, probability = predict.predict_from_data(
                model, scaler, feature_names, encoded_input
            )
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")
            st.stop()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

        if probability is not None:
            st.write(f"Probability: **{probability * 100:.2f}%**")
            st.progress(float(probability))

    st.markdown("---")
    st.caption("This project is for educational purposes only, not medical advice.")


if __name__ == '__main__':
    main()
