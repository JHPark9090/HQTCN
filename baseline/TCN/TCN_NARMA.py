import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

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
    # qml.numpy.random.seed(seed)                # for noise channels, etc.
    

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


# --- 2. TCN Components ---

class Chomp1d(nn.Module):
    """
    Removes the last 'chomp_size' elements from the last dimension of a tensor.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A single temporal block, which is the building block of the TCN.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    The main Temporal Convolutional Network model.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 3. TCN Model for NARMA Regression ---

class TCN_NARMA(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        super(TCN_NARMA, self).__init__()
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size, dropout)
        # The final layer maps the output of the last TCN layer to a single value
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        y = self.tcn(x)
        # We take the output from the last time step
        # y shape: (batch_size, num_channels[-1], seq_len)
        last_time_step = y[:, :, -1]
        # Map to the final output value
        output = self.linear(last_time_step)
        return output.squeeze(-1)


# --- 4. Training and Evaluation ---

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

def plot_loss(train_losses, val_losses, test_losses, filename="tcn_loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('TCN Training, Validation, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"Loss curve saved to {filename}")
    plt.show()

def plot_predictions(predictions, labels, scaler, train_size, val_size, filename="tcn_predictions.png"):
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    labels_rescaled = scaler.inverse_transform(labels.reshape(-1, 1))

    plt.figure(figsize=(15, 6))
    plt.plot(labels_rescaled, label='Ground Truth', color='blue', alpha=0.7)
    plt.plot(predictions_rescaled, label='Predictions', color='red', linestyle='--')
    
    plt.axvline(x=train_size, color='g', linestyle='--', label='Train/Val Split')
    plt.axvline(x=train_size + val_size, color='m', linestyle='--', label='Val/Test Split')
    
    plt.title('TCN NARMA Predictions vs Ground Truth')
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
    loss_csv_path = os.path.join(exp_name, "tcn_narma_losses.csv")
    df_loss.to_csv(loss_csv_path, index=False)
    print(f"Saved epoch losses to {loss_csv_path}")

    df_timeseries = pd.DataFrame(timeseries_data)
    ts_csv_path = os.path.join(exp_name, "tcn_narma_timeseries.csv")
    df_timeseries.to_csv(ts_csv_path, index=False)
    print(f"Saved final time series to {ts_csv_path}")
    

# --- 6. Main Execution ---

if __name__ == '__main__':
    # Hyperparameters
    N_SAMPLES = 240
    ORDER = 10
    SEQ_LEN = 10
    BATCH_SIZE = 32
    # Defines the number of output channels for each TCN layer
    NUM_CHANNELS = [32, 32, 32] 
    KERNEL_SIZE = 2
    DROPOUT = 0.3
    EPOCHS = 50
    SEED = 2027
    EXP_NAME = f"TCN_NARMA_Experiment_{SEED}"

    # Set seed
    set_all_seeds(seed = SEED)

    # Load data
    train_loader, val_loader, test_loader, input_dim, scaler, full_dataset, train_size, val_size = get_narma_dataloaders(
       
    )
    
    # Initialize model
    # input_dim has shape (batch, channels, seq_len)
    input_channels = input_dim[1]
    print(f"Input dim: {input_dim}")    # (32, 1, 20)
    
    model = TCN_NARMA(
        input_channels=input_channels,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}  |  Trainable: {trainable_params:,}  |  Frozen: {total_params-trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4, eps=1e-8)

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
            torch.save(model.state_dict(), 'tcn_narma_best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}")

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Val. Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f}')

    # Load best model for final predictions
    model.load_state_dict(torch.load('tcn_narma_best_model.pth'))
    
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