import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_and_process_data(file_path="train.csv"):
    df = pd.read_csv(file_path)
    df = df.dropna()  # Missing values hatao
    X = df.drop(columns=['Price ($)'])  # Features
    y = df['Price ($)']  # Target
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_price(model, user_input, feature_columns):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_df = pd.get_dummies(input_df)
    X_dummy, _ = load_and_process_data()
    input_df = input_df.reindex(columns=X_dummy.columns, fill_value=0)
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    X, y = load_and_process_data()
    model = train_model(X, y)
    user_input = {
        'Brand': 'OnePlus',
        'Processor': 'Dimensity 9200',
        'RAM (GB)': 8,
        'Storage (GB)': 64,
        'Camera (MP)': 108,
        'Battery (mAh)': 5000,
        'Display Size (inches)': 6.7,
        'Refresh Rate (Hz)': 144,
        '5G Support': 1,
        'Operating System': 'Android',
        'Launch Year': 2021
    }
    feature_columns = ['Brand', 'Processor', 'RAM (GB)', 'Storage (GB)', 'Camera (MP)', 
                       'Battery (mAh)', 'Display Size (inches)', 'Refresh Rate (Hz)', 
                       '5G Support', 'Operating System', 'Launch Year']
    price = predict_price(model, user_input, feature_columns)
    print(f"Predicted Price: {price:.2f}")