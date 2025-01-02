import pandas as pd
import os

def save_cluster_descriptions(stock_data, numerical_features, output_folder='./results/clusters/'):
    """Generate and save descriptions for each cluster."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Group data by cluster
    cluster_summary = stock_data.groupby('Cluster')[numerical_features].mean()

    for cluster_id, cluster_data in cluster_summary.iterrows():
        cluster_description = (
            f"Cluster {cluster_id} Summary:\n"
            f"- Average Stock Price: {cluster_data['Current Price']:.2f}\n"
            f"- Average Price Change: {cluster_data['Price Change']:.2f}\n"
            f"- Average Volatility: {cluster_data['Volatility']:.2f}\n"
            f"- Average ROE: {cluster_data['ROE']:.2f}\n"
            f"- Average Cash Ratio: {cluster_data['Cash Ratio']:.2f}\n"
            f"- Other Indicators:\n"
            f"  - Net Cash Flow: {cluster_data['Net Cash Flow']:.2f}\n"
            f"  - EPS: {cluster_data['Earnings Per Share']:.2f}\n"
            f"  - P/E Ratio: {cluster_data['P/E Ratio']:.2f}\n\n"
        )

        # Define the characteristics of this cluster
        if cluster_data['ROE'] > 15 and cluster_data['Volatility'] < 0.2:
            cluster_description += "This cluster represents stable, high-profitability companies suitable for low-risk investment.\n"
        elif cluster_data['Volatility'] > 0.4:
            cluster_description += "This cluster represents high-volatility, potentially high-growth companies suitable for risk-tolerant investors.\n"
        elif cluster_data['P/E Ratio'] < 15:
            cluster_description += "This cluster represents value stocks with low valuation metrics.\n"
        else:
            cluster_description += "This cluster includes mixed companies with diverse financial profiles.\n"

        # Save to a file
        file_path = os.path.join(output_folder, f'cluster_{cluster_id}_description.txt')
        with open(file_path, 'w') as file:
            file.write(cluster_description)

    print("\nAll cluster descriptions have been generated and saved.")

    return cluster_summary


def analyze_clusters(stock_data, numerical_features):
    """Analyze and summarize cluster characteristics."""
    cluster_summary = stock_data.groupby('Cluster')[numerical_features].mean()
    print("Cluster Summary:\n", cluster_summary)
    return cluster_summary
def save_clustered_data(data):
    """Save the clustered data to a CSV file."""
    os.makedirs(os.path.dirname('./results/clustered_data.csv'), exist_ok=True)
    data.to_csv('./results/clustered_data.csv', index=False)


if __name__ == "__main__":
    from preprocess import load_data, preprocess_data
    from clustering import apply_kmeans

    file_path = '../data/stock_data.csv'
    numerical_features = ['Current Price', 'Price Change', 'Volatility', 'ROE', 'Cash Ratio',
                          'Net Cash Flow', 'Net Income', 'Earnings Per Share', 'P/E Ratio', 'P/B Ratio']

    # Load and preprocess data
    stock_data = load_data(file_path)
    normalized_data, scaler = preprocess_data(stock_data, numerical_features)

    # Apply KMeans clustering
    clusters, kmeans_model = apply_kmeans(normalized_data, n_clusters=4)
    stock_data['Cluster'] = clusters

    # Analyze clusters
    cluster_summary = analyze_clusters(stock_data, numerical_features)
    cluster_summary.to_csv('../results/cluster_summary.csv', index=True)
    print("Cluster analysis saved!")


def recommend_stocks(data, cluster_column, preferred_cluster, top_n=5):
    """Recommend top N stocks from the preferred cluster."""

    
    cluster_data = data[data[cluster_column] == preferred_cluster]
    ranked_stocks = cluster_data.sort_values(by='ROE', ascending=False).head(top_n)
    return ranked_stocks

