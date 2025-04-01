import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend import load_and_process_data, train_model

st.title("Price Prediction App")

# File upload option
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Data load aur process
    X, y = load_and_process_data(uploaded_file)
    model, X_test, y_test, y_pred, mae, mse, r2 = train_model(X, y)

    # Results display
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2 * 100:.2f}%")

    # Predicted vs Actual Plot
    st.write("### Predicted vs Actual Prices")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to see predictions!")