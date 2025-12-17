import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Salary Prediction - Decision Tree",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Salary Prediction App")
st.write("Decision Tree Regression based Salary Prediction")

# ===============================
# Load dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Salary Data.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ===============================
# Drop missing salary values
# ===============================
data = data.dropna(subset=['Salary'])

# ===============================
# Encode categorical columns
# ===============================
label_encoders = {}
categorical_columns = ['Gender', 'Education Level', 'Job Title']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# ===============================
# Features & Target
# ===============================
X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = data['Salary']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Decision Tree Model
# ===============================
@st.cache_resource
def train_model(X_train, y_train):
    model = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ===============================
# Model Evaluation
# ===============================
y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

# ===============================
# User Input Section
# ===============================
st.subheader("🔢 Enter Employee Details")

age = st.number_input("Age", min_value=18, max_value=65, value=28)

gender = st.selectbox(
    "Gender",
    label_encoders['Gender'].classes_
)

education = st.selectbox(
    "Education Level",
    label_encoders['Education Level'].classes_
)

job_title = st.selectbox(
    "Job Title",
    label_encoders['Job Title'].classes_
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=5
)

# ===============================
# Prediction
# ===============================
if st.button("💰 Predict Salary"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [label_encoders['Gender'].transform([gender])[0]],
        'Education Level': [label_encoders['Education Level'].transform([education])[0]],
        'Job Title': [label_encoders['Job Title'].transform([job_title])[0]],
        'Years of Experience': [experience]
    })

    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

# ===============================
# Decision Tree Visualization
# ===============================
st.subheader("🌳 Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    ax=ax
)
st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built with Streamlit & Decision Tree Regression")
