import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Custom CSS for enhanced styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%; /* Make buttons full width in their column */
        margin-top: 10px; /* Add some space above buttons */
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .stButton>button:active {
        background-color: #3e8e41;
        transform: translateY(0);
        box-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    h2, h3, h4 {
        color: #34495e;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        font-weight: 600;
    }
    .stInfo { background-color: #e6f7ff; color: #0050b3; border: 1px solid #91d5ff; }
    .stSuccess { background-color: #f6ffed; color: #237804; border: 1px solid #b7eb8f; }
    .stWarning { background-color: #fffbe6; color: #ad6800; border: 1px solid #ffe58f; }
    .stError { background-color: #fff0f6; color: #a8071a; border: 1px solid #ffadd2; }

    /* Centering the main content on wide screens */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .css-18e3th9 { /* Target the main content wrapper for max width */
        max-width: 800px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- 1. Load the Trained Model ---
model_filename = 'tuned_xgboost_salary_predictor.joblib'
try:
    model = joblib.load(model_filename)
    # st.sidebar.success(f"Model loaded successfully from '{model_filename}'!") # Removed this line
except FileNotFoundError:
    st.sidebar.error(f"Error: '{model_filename}' not found.")
    st.sidebar.info("Please ensure the model file is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Define Feature Lists (Crucial for consistent input order and options) ---
WORKCLASS_OPTIONS = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
                     'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
EDUCATION_OPTIONS = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                     'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                     '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
MARITAL_STATUS_OPTIONS = ['Married-civ-spouse', 'Divorced', 'Never-married',
                          'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
OCCUPATION_OPTIONS = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                      'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
RELATIONSHIP_OPTIONS = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
RACE_OPTIONS = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
GENDER_OPTIONS = ['Male', 'Female']
NATIVE_COUNTRY_OPTIONS = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada',
                          'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica',
                          'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam',
                          'Guatemala', 'Japan', 'Poland', 'Columbia', 'Haiti', 'Portugal',
                          'Taiwan', 'Iran', 'Greece', 'Nicaragua', 'Peru', 'Ecuador', 'France',
                          'Ireland', 'Hong', 'Thailand', 'Cambodia', 'Laos', 'Yugoslavia',
                          'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland',
                          'Holand-Netherlands', 'Other']

FEATURE_COLUMNS_ORDER = [
    'age', 'workclass', 'education', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country',
    'has_capital_gain', 'has_capital_loss'
]

# --- 3. Streamlit App Interface ---
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💰 Employee Salary Predictor")
st.markdown("---")
st.markdown("Enter employee details to get an **estimated annual salary** based on predicted income bracket.")
st.markdown("---")

# Use st.form to group inputs and control re-runs
with st.form("employee_details_form"):
    st.header("Employee Details")
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 17, 90, 30, key="age_input")
            workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS, key="workclass_input")
            education = st.selectbox("Education", EDUCATION_OPTIONS, key="education_input")
            educational_num = st.slider("Educational Num", 1, 16, 10, help="Years of education or equivalent level", key="educational_num_input")
            marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS, key="marital_status_input")
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, key="occupation_input")
            relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS, key="relationship_input")

        with col2:
            race = st.selectbox("Race", RACE_OPTIONS, key="race_input")
            gender = st.selectbox("Gender", GENDER_OPTIONS, key="gender_input")
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0, help="Amount of capital gains", key="capital_gain_input")
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0, help="Amount of capital losses", key="capital_loss_input")
            hours_per_week = st.slider("Hours per Week", 1, 99, 40, key="hours_per_week_input")
            native_country = st.selectbox("Native Country", NATIVE_COUNTRY_OPTIONS, key="native_country_input")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_button = st.form_submit_button("Predict Salary 🚀") # Changed button text and added emoji
    with col_btn2:
        clear_button = st.form_submit_button("Clear Form 🔄") # Added Clear Form button

# --- Prediction Logic ---
if predict_button:
    st.subheader("Prediction Result:")
    with st.spinner('Calculating estimated salary...'):
        try:
            user_input = {
                'age': age, 'workclass': workclass, 'education': education,
                'educational-num': educational_num, 'marital-status': marital_status,
                'occupation': occupation, 'relationship': relationship,
                'race': race, 'gender': gender, 'capital-gain': capital_gain,
                'capital-loss': capital_loss, 'hours-per-week': hours_per_week,
                'native-country': native_country
            }

            user_input['has_capital_gain'] = 1 if user_input['capital-gain'] > 0 else 0
            user_input['has_capital_loss'] = 1 if user_input['capital-loss'] > 0 else 0

            input_df = pd.DataFrame([user_input])
            input_df = input_df[FEATURE_COLUMNS_ORDER]

            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            if prediction == 0: # Predicted <=50K
                estimated_salary = "$35,000"
                salary_range = "$20,000 - $50,000"
                st.info(f"The model predicts an income bracket of **<=50K**.")
                st.markdown(f"### **Estimated Annual Salary: {estimated_salary}**")
                st.write(f"*(Typically ranges from {salary_range})*")
                # st.warning("❗ Please note: This is an **estimate** based on the predicted income bracket, not an exact salary prediction from a regression model.")
            else: # Predicted >50K
                estimated_salary = "$75,000"
                salary_range = "$50,000 - $100,000+"
                st.success(f"The model predicts an income bracket of **>50K**.")
                st.markdown(f"### **Estimated Annual Salary: {estimated_salary}**")
                st.write(f"*(Typically ranges from {salary_range})*")
                # st.warning("❗ Please note: This is an **estimate** based on the predicted income bracket, not an exact salary prediction from a regression model.")

            st.markdown("---")
            st.write(f"Probability of <=50K: {prediction_proba[0]:.2f}")
            st.write(f"Probability of >50K: {prediction_proba[1]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and ensure the model loaded correctly.")

# --- Clear Form Logic ---
elif clear_button:
    # This will trigger a rerun and reset the form inputs to their default values
    # by using session state to force default values on next run
    for key in st.session_state.keys():
        if key.endswith("_input"): # Target input widgets by their keys
            del st.session_state[key]
    st.experimental_rerun() # Force a rerun to clear inputs

st.markdown("---")

# --- How to Use Section ---
with st.expander("❓ How to Use This Predictor"):
    st.markdown("""
    1.  **Enter Employee Details:** Adjust the sliders, select boxes, and number inputs in the "Employee Details" section.
        * Provide information like age, workclass, education, marital status, occupation, and financial details.
    2.  **Click 'Predict Salary':** Once all details are entered, click the "Predict Salary 🚀" button.
    3.  **View Results:** The estimated annual salary and the predicted income bracket will be displayed below.
    4.  **Clear Form:** Click "Clear Form 🔄" to reset all input fields to their default values.
    """)

# --- About Section ---
# with st.expander("ℹ️ About This Predictor"):
#     st.markdown("""
#     This application estimates an employee's annual salary based on various demographic and employment features.
#     It uses a machine learning model (XGBoost Classifier) trained on a publicly available dataset.

#     **Important Note:** The underlying model predicts whether an income is less than or equal to $50,000 (<=50K) or greater than $50,000 (>50K).
#     The "Estimated Annual Salary" displayed is a representative value for the predicted income bracket, not an exact numerical prediction.
#     """)

st.markdown("Built with ❤️ by TAPAN")
