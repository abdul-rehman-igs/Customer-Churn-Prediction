import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

model = joblib.load('churn_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')
df = pd.read_csv("Telco-Customer-Churn.csv")

st.sidebar.header("🔍 Enter Customer Data")

def user_input_features():
    gender = st.sidebar.radio("Gender", ['Male', 'Female'])
    SeniorCitizen = st.sidebar.radio("Senior Citizen", ['Yes', 'No'])
    Partner = st.sidebar.radio("Partner", ['Yes', 'No'])
    Dependents = st.sidebar.radio("Dependents", ['Yes', 'No'])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 24)
    PhoneService = st.sidebar.radio("Phone Service", ['Yes', 'No'])
    PaperlessBilling = st.sidebar.radio("Paperless Billing", ['Yes', 'No'])
    InternetService = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    Contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    PaymentMethod = st.sidebar.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
    TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    data = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'PaperlessBilling': PaperlessBilling,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return pd.DataFrame([data])

input_df = user_input_features()

def preprocess_input(df):
    df = df.copy()
    df.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    return df[model_columns]

processed_input = preprocess_input(input_df)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA", "📂 Raw Data"])

with tab1:
    st.subheader("🧠 Churn Prediction")
    if st.button("Predict Churn"):
        prediction = model.predict(processed_input)[0]
        proba = model.predict_proba(processed_input)[0][1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", "Churn" if prediction == 1 else "Not Churn", delta=None)
        with col2:
            st.metric("Churn Probability", f"{proba:.2%}")
        
        st.success("The customer is likely to **CHURN**." if prediction == 1 else "The customer is likely to **STAY**.")

with tab2:
    st.subheader("📈 Exploratory Data Analysis")
    df_cleaned = df.dropna().copy()
    df_cleaned['Churn'] = df_cleaned['Churn'].replace({'Yes': 1, 'No': 0})

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Churn Count**")
        fig = px.histogram(df, x='Churn', color='Churn', color_discrete_sequence=['skyblue', 'salmon'])
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("**Monthly Charges Distribution**")
        fig = px.box(df, x='Churn', y='MonthlyCharges', color='Churn', color_discrete_sequence=['skyblue', 'salmon'])
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📂 Raw Telco Customer Data")
    st.dataframe(df.head(50), use_container_width=True)
