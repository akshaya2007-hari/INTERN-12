import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ===============================
# App Title
# ===============================
st.title("🏠 House Price Prediction App")
st.write("This app predicts house prices using a Random Forest Regression model.")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("house_prices_dataset.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ===============================
# Features & Target
# ===============================
X = data[['square_feet', 'num_rooms', 'age', 'distance_to_city(km)']]
y = data['price']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Random Forest Model
# ===============================
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ===============================
# Model Evaluation
# ===============================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# ===============================
# User Input Section
# ===============================
st.subheader("🔢 Enter House Details")

square_feet = st.number_input(
    "Square Feet",
    min_value=300,
    max_value=10000,
    value=1500
)

num_rooms = st.number_input(
    "Number of Rooms",
    min_value=1,
    max_value=20,
    value=3
)

age = st.number_input(
    "House Age (years)",
    min_value=0,
    max_value=100,
    value=10
)

distance = st.number_input(
    "Distance to City (km)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

# ===============================
# Prediction
# ===============================
if st.button("💰 Predict House Price"):
    input_data = pd.DataFrame({
        'square_feet': [square_feet],
        'num_rooms': [num_rooms],
        'age': [age],
        'distance_to_city(km)': [distance]
    })

    prediction = model.predict(input_data)

    st.success(f"🏷️ Predicted House Price: ₹ {prediction[0]:,.2f}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("📌 Built using Streamlit & Random Forest Regression")
