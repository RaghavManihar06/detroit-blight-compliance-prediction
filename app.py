import streamlit as st
import pandas as pd
import joblib

model = joblib.load('blight_compliance_model.pkl')
features = joblib.load('model_features.pkl')
threshold = joblib.load('model_threshold.pkl')

st.title("Detroit Blight Ticket Compliance Predictor")
st.write("Fill in ticket details to predict if it will be paid on time.")

# these are the one-hot groups from training, pull the categories straight from the saved feature list
# so the dropdowns always match whatever the model was actually trained on
prefixes = ['Ordinance Description_', 'Disposition_', 'Agency Name_', 'Neighborhood_']

grouped_options = {}
for prefix in prefixes:
    options = [c[len(prefix):] for c in features if c.startswith(prefix)]
    grouped_options[prefix] = ['(baseline / other)'] + sorted(options)

col1, col2 = st.columns(2)

with col1:
    fine_amount = st.number_input("Fine Amount ($)", min_value=0, value=250)
    discount_amount = st.number_input("Discount Amount ($)", min_value=0, value=0)
    days_to_hearing = st.number_input("Days to Hearing", min_value=0, max_value=1000, value=52)
    council_district = st.number_input("Council District", min_value=1, max_value=10, value=5)

with col2:
    latitude = st.number_input("Latitude", value=42.35, format="%.6f")
    longitude = st.number_input("Longitude", value=-83.05, format="%.6f")
    is_out_of_state = st.checkbox("Owner is out of state")
    owner_in_detroit = st.checkbox("Owner lives in Detroit")
    owner_at_property = st.checkbox("Owner lives at the violation property")

st.subheader("Violation details")

selected = {}
labels = {
    'Ordinance Description_': "Ordinance Description",
    'Disposition_': "Disposition",
    'Agency Name_': "Agency Name",
    'Neighborhood_': "Neighborhood",
}
for prefix, label in labels.items():
    selected[prefix] = st.selectbox(label, grouped_options[prefix])

if st.button("Predict Compliance"):
    row = pd.DataFrame([[0] * len(features)], columns=features)

    row['Fine Amount'] = fine_amount
    row['Discount Amount'] = discount_amount
    row['days_to_hearing'] = days_to_hearing
    row['Council District'] = council_district
    row['Latitude'] = latitude
    row['Longitude'] = longitude
    row['is_out_of_state_owner'] = is_out_of_state
    row['owner_lives_in_detroit'] = owner_in_detroit
    row['owner_lives_at_violation_property'] = owner_at_property

    for prefix, choice in selected.items():
        if choice != '(baseline / other)':
            col_name = prefix + choice
            if col_name in row.columns:
                row[col_name] = 1

    proba = model.predict_proba(row[features])[:, 1][0]
    prediction = "Compliant (likely to pay)" if proba >= threshold else "High risk / non-compliant"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Compliance probability: {proba:.1%}")
    st.progress(min(max(float(proba), 0.0), 1.0))