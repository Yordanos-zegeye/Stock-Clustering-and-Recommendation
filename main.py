from src.preprocess import load_data, preprocess_data
from src.clustering import elbow_method, apply_kmeans
from src.analysis import analyze_clusters, recommend_stocks, save_cluster_descriptions, save_clustered_data
import os


if __name__ == "__main__":
    file_path = './data/stock_data.csv'
    numerical_features = ['Current Price', 'Price Change', 'Volatility', 'ROE', 'Cash Ratio',
                          'Net Cash Flow', 'Net Income', 'Earnings Per Share', 'P/E Ratio', 'P/B Ratio']

    # Load and preprocess the data
    stock_data = load_data(file_path)
    normalized_data, scaler = preprocess_data(stock_data, numerical_features)

    # Determine the optimal number of clusters using the Elbow Method
    elbow_method(normalized_data)

    # Apply KMeans clustering
    clusters, kmeans_model = apply_kmeans(normalized_data, n_clusters=4)
    stock_data['Cluster'] = clusters

    # Analyze the clusters
    cluster_summary = analyze_clusters(stock_data, numerical_features)
    print("Clustering and analysis pipeline complete!")
    cluster_description = save_cluster_descriptions(stock_data , numerical_features)
    # Load clusterd data to a new file
    cluster_column = 'Cluster' 
    if cluster_column in stock_data.columns:
        save_clustered_data(stock_data)

    # Recommend stocks from Cluster 1 (e.g., high ROE and EPS)
    output_folder = './results/recommendations/'
    os.makedirs(output_folder, exist_ok=True)

    clusters = stock_data['Cluster'].unique()
    top_n = 5 

for cluster_id in clusters:
    recommended_stocks = recommend_stocks(stock_data, 'Cluster', cluster_id, top_n=top_n)

    # Save recommendations to a file
    output_file = os.path.join(output_folder, f'recommended_stocks_cluster_{cluster_id}.csv')
    recommended_stocks.to_csv(output_file, index=False)
