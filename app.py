import streamlit as st
from backend import load_and_process_data, train_model, predict_price

st.title("Price Prediction App")

# Model ko pehle train karo
X, y = load_and_process_data()  # Dummy data ya CSV se
model = train_model(X, y)

# User input fields
st.write("### Enter Details to Predict Price")
category = st.selectbox("Category", ['A', 'B', 'C'])
size = st.number_input("Size", min_value=0, max_value=100, value=10)
condition = st.selectbox("Condition", ['Good', 'Bad'])

# User input ko dictionary mein daalo
user_input = {
    'Category': category,
    'Size': size,
    'Condition': condition
}
feature_columns = ['Category', 'Size', 'Condition']

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(model, user_input, feature_columns)
    st.write(f"### Predicted Price: ${predicted_price:.2f}")

st.write("Enter values and click 'Predict Price' to see the result!")