# Stock Clustering Analysis Project

## Project Structure

- `data/`: Contains the stock data file (`stock_data.csv`).
- `src/`: Python scripts for preprocessing, clustering, and analysis.
- `notebooks/`: Jupyter notebook for exploratory analysis.
- `results/`: Stores clustering results and plots.
- `main.py`: Orchestrates the entire clustering pipeline.

## Libraries Used

- Pandas
- scikit-learn
- matplotlib
- numpy
- seaborn

## Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Install dependencies
3. Place your dataset in the `data/` folder as `stock_data.csv`.
4. Run the project by executing `python main.py`.

## Project Workflow

1. **Preprocessing**: The `preprocess.py` script normalizes the numerical data.
2. **Clustering**: The `clustering.py` script applies KMeans clustering.
3. **Analysis**: The `analysis.py` script summarizes and saves cluster characteristics.

## Clustering Methodology

We use the Elbow Method to determine the optimal number of clusters and apply KMeans for segmentation of stocks based on financial indicators.
