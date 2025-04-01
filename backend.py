import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Price ($)'])  # Features
    y = df['Price ($)']  # Target
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X_test, y_test, y_pred, mae, mse, r2

if __name__ == "__main__":
    X, y = load_and_process_data("train.csv")  # Apna dataset daal
    model, X_test, y_test, y_pred, mae, mse, r2 = train_model(X, y)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2 * 100:.2f}%")