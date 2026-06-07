import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost # Required for loading XGBoost model

# --- Model and Feature Loading (inside the Streamlit app) ---

model_filename = 'best_model(1) (1).pkl'

try:
    with open(model_filename, 'rb') as file:
        best_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Streamlit App Error: Model file '{model_filename}' not found.")
    st.stop() # Stop the Streamlit app if model isn't found
except Exception as e:
    st.error(f"Streamlit App Error: An error occurred while loading the model: {e}")
    st.stop()

feature_order = [
    'app_usage_time_min',
    'swipe_right_ratio',
    'likes_received',
    'mutual_matches',
    'income_bracket_Low',
    'income_bracket_Lower-Middle',
    'income_bracket_Middle',
    'income_bracket_Upper-Middle',
    'income_bracket_Very High',
    'income_bracket_Very Low',
    'education_level_Bachelor’s',
    'education_level_Diploma',
    'education_level_High School',
    'education_level_MBA',
    'education_level_Master’s',
    'education_level_No Formal Education',
    'education_level_PhD',
    'education_level_Postdoc'
]

# Streamlit App Interface
st.title('User Churn Prediction Dashboard')
st.write('Adjust the parameters below to predict whether a user will churn.')

# Create Input Controls
user_input_values = {}

# Numerical Features Input
user_input_values['app_usage_time_min'] = st.slider(
    'App Usage Time (minutes)',
    min_value=0.0,
    max_value=1500.0,
    value=500.0,
    step=1.0
)
user_input_values['swipe_right_ratio'] = st.slider(
    'Swipe Right Ratio (0.0 - 1.0)',
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)
user_input_values['likes_received'] = st.number_input(
    'Likes Received',
    min_value=0,
    max_value=1000,
    value=50,
    step=1
)
user_input_values['mutual_matches'] = st.number_input(
    'Mutual Matches',
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

# Categorical Features Input (One-Hot Encoded)
income_bracket_options = [
    'Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'Very High'
]
selected_income_bracket = st.selectbox(
    'Income Bracket',
    options=income_bracket_options,
    index=income_bracket_options.index('Middle') # Default value
)

education_level_options = [
    'No Formal Education', 'High School', 'Diploma', 'Bachelor’s', 'Master’s', 'MBA', 'PhD', 'Postdoc'
]
selected_education_level = st.selectbox(
    'Education Level',
    options=education_level_options,
    index=education_level_options.index('Bachelor’s') # Default value
)

# Prediction Button
if st.button('Predict Churn'):
    # Convert user input to DataFrame and ensure feature order matches model training
    # Initialize a DataFrame with all zeros, using feature_order as column names
    input_data = {feature: [0] for feature in feature_order}

    # Fill numerical features
    for feature in ['app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches']:
        input_data[feature] = [user_input_values[feature]]

    # Fill one-hot encoded income bracket feature
    income_feature_name = f'income_bracket_{selected_income_bracket}'
    if income_feature_name in input_data:
        input_data[income_feature_name] = [1]

    # Fill one-hot encoded education level feature
    education_feature_name = f'education_level_{selected_education_level}'
    if education_feature_name in input_data:
        input_data[education_feature_name] = [1]

    final_input_df = pd.DataFrame(input_data)

    # Ensure DataFrame column order matches feature_order
    final_input_df = final_input_df[feature_order]

    # Make prediction
    prediction = best_model.predict(final_input_df)
    prediction_proba = best_model.predict_proba(final_input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'This user has a high risk of churn! (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'This user has a low risk of churn. (Probability: {prediction_proba[0]:.2f})')
