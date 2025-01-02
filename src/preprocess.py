import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data, numerical_features):
    """Normalize numerical data for clustering."""
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[numerical_features])
    return normalized_data, scaler

if __name__ == "__main__":
    # Define the file path and numerical features
    file_path = '../data/stock_data.csv'
    numerical_features = ['Current Price', 'Price Change', 'Volatility', 'ROE', 'Cash Ratio',
                          'Net Cash Flow', 'Net Income', 'Earnings Per Share', 'P/E Ratio', 'P/B Ratio']
    
    # Load and preprocess the data
    stock_data = load_data(file_path)
    normalized_data, scaler = preprocess_data(stock_data, numerical_features)
    print("Data Preprocessed!")
