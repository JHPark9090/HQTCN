import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def generate_narma_data(n_samples, order, seed=None):
    """
    Generates NARMA time-series data.
    """
    # Fix the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    u = np.random.uniform(0, 0.5, n_samples)
    y = np.zeros(n_samples)

    for t in range(order, n_samples):
        term1 = 0.3 * y[t-1]
        term2 = 0.05 * y[t-1] * np.sum(y[t-i-1] for i in range(order))
        term3 = 1.5 * u[t-order] * u[t-1]
        term4 = 0.1
        y[t] = term1 + term2 + term3 + term4
        
    return y.reshape(-1, 1)

def transform_narma_data(data, seq_length):
    """
    Transforms NARMA data into input-output pairs for sequence prediction.
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    x = torch.from_numpy(np.array(x)).float()
    y = torch.from_numpy(np.array(y)).float()
    
    return x, y

def get_narma_data(n_samples=240, order=10, seq_length=10, seed=None):
    """
    Generates and transforms NARMA data for the QLSTM model.
    """
    seed = seed
    print(f"seed: {seed}")

    # Generate NARMA data
    narma_series = generate_narma_data(n_samples, order, seed=seed)

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(narma_series)

    # Transform data into sequences
    x, y = transform_narma_data(dataset, seq_length)
    
    return x, y

if __name__ == '__main__':
    # Example of how to use the functions
    x_data, y_data = get_narma_data()
    print("Shape of X data:", x_data.shape)
    print("Shape of Y data:", y_data.shape)
