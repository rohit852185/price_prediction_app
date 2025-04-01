import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dummy dataset banaya agar CSV nahi hai
def create_dummy_data():
    data = {
        'Category': ['A', 'B', 'C', 'A', 'B'],
        'Size': [10, 15, 20, 12, 18],
        'Condition': ['Good', 'Bad', 'Good', 'Bad', 'Good'],
        'Price ($)': [500, 300, 700, 450, 600]
    }
    return pd.DataFrame(data)

def load_and_process_data(file_path=None):
    if file_path:
        df = pd.read_csv(file_path)
    else:
        df = create_dummy_data()  # Agar file nahi hai toh dummy data
    df = df.dropna()
    X = df.drop(columns=['Price ($)'])
    y = df['Price ($)']
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model  # Sirf trained model return karo

def predict_price(model, user_input, feature_columns):
    # User input ko DataFrame mein convert karo
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    # Categorical columns ko one-hot encode karo
    input_df = pd.get_dummies(input_df)
    # Model ke training data ke columns se match karo
    X_dummy, _ = load_and_process_data()
    input_df = input_df.reindex(columns=X_dummy.columns, fill_value=0)
    # Prediction karo
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    X, y = load_and_process_data()
    model = train_model(X, y)
    # Test prediction
    user_input = {'Category': 'A', 'Size': 10, 'Condition': 'Good'}
    feature_columns = ['Category', 'Size', 'Condition']
    price = predict_price(model, user_input, feature_columns)
    print(f"Predicted Price: {price:.2f}")