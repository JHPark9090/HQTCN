import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

import pennylane as qml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import time, os, random
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on (Python, NumPy, Torch, PennyLane, CUDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # no-op on CPU
    torch.backends.cudnn.deterministic = True  # reproducible convolutions
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)                # for noise channels, etc.
    

# --- 1. NARMA Data Generation ---

def generate_narma_data(n_samples, order, seed=None):
    """
    Generates NARMA time-series data with a fixed seed for reproducibility.
    """
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

def transform_narma_data(data, seq_len):
    """
    Transforms NARMA data into input-output pairs for sequence prediction.
    """
    x, y = [], []
    for i in range(len(data) - seq_len):
        _x = data[i:(i + seq_len)]
        _y = data[i + seq_len]
        x.append(_x)
        y.append(_y)

    # Reshape x to be (batch, channels, seq_len) for Conv1d
    x = np.array(x).transpose(0, 2, 1) 
    
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(np.array(y)).float()
    
    return x, y

def get_narma_dataloaders(n_samples=2000, order=10, seq_len=20, batch_size=32, train_p=0.7, val_p=0.15, seed=None):
    """
    Generates and transforms NARMA data, then creates DataLoader objects using sequential splitting.
    """
    print("Generating NARMA data...")
    narma_series = generate_narma_data(n_samples, order, seed=seed)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_scaled = scaler.fit_transform(narma_series)

    x, y = transform_narma_data(dataset_scaled, seq_len)
    
    full_dataset = TensorDataset(x, y)

    # Sequential split
    train_end_idx = int(train_p * len(full_dataset))
    val_end_idx = int((train_p + val_p) * len(full_dataset))
    
    train_indices = list(range(train_end_idx))
    val_indices = list(range(train_end_idx, val_end_idx))
    test_indices = list(range(val_end_idx, len(full_dataset)))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Data loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    input_dim = (batch_size, x.shape[1], x.shape[2]) # (batch, channels, seq_len)

    return train_loader, val_loader, test_loader, input_dim, scaler, full_dataset, len(train_dataset), len(val_dataset)



# --- 2. QTCN Model ---
class QTCN(nn.Module):
    def __init__(self, n_qubits, circuit_depth, input_dim, kernel_size, dilation=1):
        super(QTCN, self).__init__()
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        # Quantum parameters
        self.conv_params = nn.Parameter(torch.randn(circuit_depth, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(circuit_depth, n_qubits // 2, 3))
        # Quantum device initialization
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qc = qml.QNode(self.circuit, self.dev)
        
        # The kernel size defines how many time steps we consider for the "convolution"
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # The input channels are treated as the feature size for each time step
        # Fully connected classical linear layer
        self.fc = nn.Linear(self.input_channels * self.kernel_size, n_qubits)  # For dimension reduction
        self.downsample = nn.Linear(self.input_channels * self.kernel_size, n_qubits)
        # self.fc_out = nn.Linear(self.time_steps - self.dilation * (self.kernel_size - 1), 1)  # Final output layer for Binary Classification

    def circuit(self, features):
        wires = list(range(self.n_qubits))    
        # Variational Embedding (Angle Embedding)
        qml.AngleEmbedding(features, wires=wires, rotation='Y')
        for layer in range(self.circuit_depth):
            # Convolutional Layer
            self._apply_convolution(self.conv_params[layer], wires)
            # Pooling Layer
            self._apply_pooling(self.pool_params[layer], wires)
            wires = wires[::2]  # Retain every second qubit after pooling
        # Measurement
        return qml.expval(qml.PauliZ(self.n_qubits-1))

    def _apply_convolution(self, weights, wires):
        """
        Convolutional layer logic (same as original).
        """
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])

    def _apply_pooling(self, pool_weights, wires):
        # Pooling using a variational circuit
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])
                
    def forward(self, x):
        # x has shape (batch_size, time_steps, input_channels)
        batch_size, input_channels, time_steps = x.size()
        # Quantum Circuit Execution

        # Initialize an empty list to store the output
        output = []
        # Slide a window of size `kernel_size` across the time steps (with dilation)
        for i in range(self.dilation * (self.kernel_size - 1), time_steps):
            indices = [i - d*self.dilation for d in range(self.kernel_size)]
            indices.reverse()
            window = x[:, :, indices].reshape(batch_size, -1)
            reduced_window = self.fc(window)
            output.append(self.qc(reduced_window))
        # output = torch.stack(output, dim=1)
        # output = self.fc_out(output.float()).squeeze(1)
        output = torch.mean(torch.stack(output, dim=1), dim=1).float()
        return output


# --- 3. Training and Evaluation ---

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            total_loss += loss.item()
            
    return total_loss / len(dataloader)
    
def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# --- 4. Plotting ---

def plot_loss(train_losses, val_losses, test_losses, filename="hqtcn2_loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('HQTCN2 Training, Validation, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Loss curve saved to {filename}")
    plt.show()

def plot_predictions(predictions, labels, scaler, train_size, val_size, filename="hqtcn2_predictions.png"):
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    labels_rescaled = scaler.inverse_transform(labels.reshape(-1, 1))

    plt.figure(figsize=(15, 6))
    plt.plot(labels_rescaled, label='Ground Truth', color='blue', alpha=0.7)
    plt.plot(predictions_rescaled, label='Predictions', color='red', linestyle='--')
    
    plt.axvline(x=train_size, color='g', linestyle='--', label='Train/Val Split')
    plt.axvline(x=train_size + val_size, color='m', linestyle='--', label='Val/Test Split')
    
    plt.title('HQTCN2 NARMA Predictions vs Ground Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Prediction plot saved to {filename}")
    plt.show()

def save_log_to_csv(exp_name, epoch_data, timeseries_data):
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    df_loss = pd.DataFrame(epoch_data)
    loss_csv_path = os.path.join(exp_name, "hqtcn2_narma_losses.csv")
    df_loss.to_csv(loss_csv_path, index=False)
    print(f"Saved epoch losses to {loss_csv_path}")

    df_timeseries = pd.DataFrame(timeseries_data)
    ts_csv_path = os.path.join(exp_name, "hqtcn2_narma_timeseries.csv")
    df_timeseries.to_csv(ts_csv_path, index=False)
    print(f"Saved final time series to {ts_csv_path}")
    

# --- 5. Main Execution ---

if __name__ == '__main__':
    # Hyperparameters
    SEED = 2027
    EXP_NAME = f"HQTCN2_NARMA_Experiment_{SEED}"
    N_QUBITS = 8
    CIRCUIT_DEPTH = 2 # Number of conv/pool layers in QCNN
    N_SAMPLES=240
    ORDER=10
    SEQ_LEN = 10
    BATCH_SIZE = 32
    KERNEL_SIZE = 5
    DILATION = 2
    EPOCHS = 50

    # Set seed
    set_all_seeds(seed = SEED)

    # Load data
    train_loader, val_loader, test_loader, input_dim, scaler, full_dataset, train_size, val_size = get_narma_dataloaders(
        n_samples=N_SAMPLES, order=ORDER, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, seed=SEED
    )
    
    # Initialize model
    model = QTCN(
        n_qubits=N_QUBITS,
        circuit_depth=CIRCUIT_DEPTH,
        input_dim=input_dim,
        kernel_size=KERNEL_SIZE,
        dilation=DILATION
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4, eps=1e-8)

    # Training loop
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        test_loss = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hqtcn2_narma_best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}")

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Val. Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}')

    # Load best model for final predictions
    model.load_state_dict(torch.load('hqtcn2_narma_best_model.pth'))
    
    # Plotting and Logging
    plot_loss(train_losses, val_losses, test_losses)
    
    print("\nGenerating predictions for the entire dataset...")
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    predictions, labels = predict(model, full_loader)
    plot_predictions(predictions, labels, scaler, train_size, val_size)

    # Save logs to CSV
    epoch_log_data = {
        'epoch': list(range(1, EPOCHS + 1)),
        'train_loss': train_losses,
        'validation_loss': val_losses,
        'test_loss': test_losses
    }
    timeseries_log_data = {
        'prediction': predictions.flatten(),
        'ground_truth': labels.flatten()
    }
    save_log_to_csv(EXP_NAME, epoch_log_data, timeseries_log_data)



