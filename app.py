import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Title
st.title("🏠 Dhaka House Price Predictor")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dhaka_housing_data_with_price.csv")
    return df

df = load_data()

# Show data and correlation heatmap
if st.checkbox("🔍 Show Dataset"):
    st.write(df.head())

if st.checkbox("📊 Show Correlation Heatmap"):
    st.write("Correlation between features:")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    import seaborn as sns
    dataset_correlation = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(dataset_correlation, cbar=True, square=True, fmt='.1f',
                annot=True, annot_kws={'size':8}, cmap='Blues')
    st.pyplot()

# Model training
X = df.iloc[:,:-1]
y = df['PRICE']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = XGBRegressor()
model.fit(x_train, y_train)

# Sidebar inputs
st.sidebar.header("Input Features")
user_input = []

for col in X.columns:
    val = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(val)

input_df = pd.DataFrame([user_input], columns=X.columns)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"🏠 Predicted House Price: {prediction[0]:,.2f} BDT")

# Evaluation (Optional)
if st.checkbox("📈 Show Model Evaluation"):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    r2_train = metrics.r2_score(y_train, train_pred)
    r2_test = metrics.r2_score(y_test, test_pred)

    mae_train = metrics.mean_absolute_error(y_train, train_pred)
    mae_test = metrics.mean_absolute_error(y_test, test_pred)

    st.write(f"🔹 Training R² Score: {r2_train:.2f}")
    st.write(f"🔹 Testing R² Score: {r2_test:.2f}")
    st.write(f"🔹 Training MAE: {mae_train:.2f}")
    st.write(f"🔹 Testing MAE: {mae_test:.2f}")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, alpha=0.5)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Price")
    st.pyplot(fig)
