import streamlit as st
from backend import load_and_process_data, train_model, predict_price

st.title("Smartphone Price Prediction App")

# Model ko train karo
X, y = load_and_process_data()
model = train_model(X, y)

# User input fields
st.write("### Enter Smartphone Details")
brand = st.selectbox("Brand", ['OnePlus', 'Samsung', 'Vivo', 'Oppo', 'Xiaomi', 'Apple', 'Realme', 'Motorola', 'Google', 'Sony'])
processor = st.selectbox("Processor", ['Dimensity 9200', 'Snapdragon 8 Gen 2', 'A17 Bionic', 'Exynos 2200', 'Kirin 9000', 'Snapdragon 778G'])
ram = st.selectbox("RAM (GB)", [4, 6, 8, 12, 16])
storage = st.selectbox("Storage (GB)", [64, 128, 256, 512, 1024])
camera = st.number_input("Camera (MP)", min_value=5, max_value=200, value=48)
battery = st.number_input("Battery (mAh)", min_value=2000, max_value=7000, value=4000)
display_size = st.number_input("Display Size (inches)", min_value=5.0, max_value=8.0, value=6.5, step=0.1)
refresh_rate = st.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144])
five_g = st.selectbox("5G Support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
os = st.selectbox("Operating System", ['Android', 'iOS'])
launch_year = st.selectbox("Launch Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024])

# User input dictionary
user_input = {
    'Brand': brand,
    'Processor': processor,
    'RAM (GB)': ram,
    'Storage (GB)': storage,
    'Camera (MP)': camera,
    'Battery (mAh)': battery,
    'Display Size (inches)': display_size,
    'Refresh Rate (Hz)': refresh_rate,
    '5G Support': five_g,
    'Operating System': os,
    'Launch Year': launch_year
}
feature_columns = ['Brand', 'Processor', 'RAM (GB)', 'Storage (GB)', 'Camera (MP)', 
                   'Battery (mAh)', 'Display Size (inches)', 'Refresh Rate (Hz)', 
                   '5G Support', 'Operating System', 'Launch Year']

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(model, user_input, feature_columns)
    st.write(f"### Predicted Price: ${predicted_price:.2f}")

st.write("Enter details and click 'Predict Price' to see the result!")