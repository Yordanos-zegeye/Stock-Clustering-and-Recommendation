from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(normalized_data, max_clusters=10):
    """Apply the Elbow Method to find the optimal number of clusters."""
    inertia = []
    range_clusters = range(1, max_clusters + 1)
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_data)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range_clusters, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

def apply_kmeans(normalized_data, n_clusters=4):
    """Apply KMeans clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(normalized_data), kmeans

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data

    file_path = '../data/stock_data.csv'
    numerical_features = ['Current Price', 'Price Change', 'Volatility', 'ROE', 'Cash Ratio',
                          'Net Cash Flow', 'Net Income', 'Earnings Per Share', 'P/E Ratio', 'P/B Ratio']

    # Load and preprocess data
    stock_data = load_data(file_path)
    normalized_data, scaler = preprocess_data(stock_data, numerical_features)

    # Determine the optimal number of clusters
    elbow_method(normalized_data)

    # Apply KMeans clustering
    clusters, kmeans_model = apply_kmeans(normalized_data, n_clusters=4)
    stock_data['Cluster'] = clusters
    print("Clustering complete!")
