import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# App Title
# -------------------------------
st.title("💼 Salary Prediction App")
st.write("Decision Tree Regression using Employee Data")

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("Salary Data.csv")
st.subheader("Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Preprocessing
# -------------------------------
label_encoders = {}
categorical_columns = ['Gender', 'Education Level', 'Job Title']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Remove missing salary rows
data = data.dropna(subset=['Salary'])

X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = data['Salary']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------
model = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

st.subheader("📊 Model Performance")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R² Score:", r2_score(y_test, y_pred))

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("🧮 Predict Salary")

age = st.number_input("Age", min_value=18, max_value=65, value=28)
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
education = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
job = st.selectbox("Job Title", label_encoders['Job Title'].classes_)
experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

# Encode Inputs
gender_enc = label_encoders['Gender'].transform([gender])[0]
education_enc = label_encoders['Education Level'].transform([education])[0]
job_enc = label_encoders['Job Title'].transform([job])[0]

input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_enc],
    'Education Level': [education_enc],
    'Job Title': [job_enc],
    'Years of Experience': [experience]
})

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")
