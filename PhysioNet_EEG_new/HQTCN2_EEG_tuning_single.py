import pennylane as qml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from typing import Tuple
from Load_PhysioNet_EEG import load_eeg_ts_revised
import math, os, copy, time, random
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=80)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--kernel_size", type=int, default=12)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=False)   # resume=True if typing --resume on terminal
    return parser.parse_args()
args=get_args()


print('Pennylane Version :', qml.__version__)
print('Pytorch Version :', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Running on ", device)


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
        self.quantum_circuit = qml.QNode(self.circuit, self.dev)
        
        # The kernel size defines how many time steps we consider for the "convolution"
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # The input channels are treated as the feature size for each time step
        # Fully connected classical linear layer
        self.fc = nn.Linear(self.input_channels * self.kernel_size, n_qubits)  # For dimension reduction

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
        return qml.expval(qml.PauliZ(0))

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
        # Initialize an empty list to store the output
        output = []
        # Slide a window of size `kernel_size` across the time steps (with dilation)
        for i in range(self.dilation * (self.kernel_size - 1), time_steps):
            indices = [i - d*self.dilation for d in range(self.kernel_size)]
            indices.reverse()
            window = x[:, :, indices].reshape(batch_size, -1)
            reduced_window = self.fc(window)
            # Quantum Circuit Execution
            output.append(self.quantum_circuit(reduced_window))
        output = torch.mean(torch.stack(output, dim=1), dim=1)
        return output


################################# Calculate Running Time ########################################
def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


################################# Performance ################################
# Training loop
def train_perf(model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    all_labels = []
    all_outputs = []
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Ensure that data is on the same device (GPU or CPU)
        labels = labels.float()   # Ensure labels are of type float for BCEWithLogitsLoss
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Collect labels and outputs for AUROC
        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())       
        
    # Calculate train AUROC
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    train_auroc = roc_auc_score(all_labels, all_outputs)
    
    return train_loss / len(dataloader), train_auroc


# Validation/Test loop
def evaluate_perf(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Ensure that data is on the same device (GPU or CPU)
            labels = labels.float()   # Ensure labels are of type float for BCEWithLogitsLoss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Collect labels and outputs for AUROC
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    auroc = roc_auc_score(all_labels, all_outputs)
    
    return running_loss / len(dataloader), auroc


def QuantumTCNN_run(seed, n_qubits, circuit_depth, input_dim, kernel_size=None, dilation=None, num_epochs=10, 
                    checkpoint_dir="QTCN_checkpoints", resume=False):
    print("Running on ", device)
    set_all_seeds(seed)
    print("Random Seed = ", seed)
    print(f"Evaluating pair: kernel_size={kernel_size}, dilation={dilation}")
    model = QTCN(n_qubits, circuit_depth, input_dim, kernel_size, dilation).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)

    # --- Checkpoint Loading Logic ---
    os.makedirs(checkpoint_dir, exist_ok=True) # <--- Create checkpoint directory if it doesn't exist
    checkpoint_path = os.path.join(checkpoint_dir, f"q_tcnn_model_freq{args.freq}_sample{args.n_sample}_lr{args.lr}_kernel{args.kernel_size}_dilation{args.dilation}_{seed}.pth")
    start_epoch = 0
    train_metrics, valid_metrics = [], [] # <--- Initialize here

    if resume==True and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_metrics = checkpoint['train_metrics']
        valid_metrics = checkpoint['valid_metrics']
        print(f"Resuming training from epoch {start_epoch + 1}")
    # --- End Checkpoint Logic ---

    # Training process
    for epoch in range(start_epoch, num_epochs): # <--- Start from the correct epoch
        start_time = time.time()
        
        train_loss, train_auc = train_perf(model, train_loader, optimizer, criterion)
        train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_auc': train_auc})
    
        valid_loss, valid_auc = evaluate_perf(model, val_loader, criterion)
        valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss, 'valid_auc': valid_auc})
    
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | Validation Loss: {valid_loss:.4f}, AUC: {valid_auc:.4f}")

        # --- Save Checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
        }, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch + 1}")
        # --- End Save Checkpoint ---

    # Final evaluation on the test set
    test_loss, test_auc = evaluate_perf(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}")
    
    # Final metrics processing
    test_metrics = [{'epoch': num_epochs, 'test_loss': test_loss, 'test_auc': test_auc}]
    
    metrics = []
    # Ensure train/valid metrics are available up to the final epoch for DataFrame creation
    final_epoch = len(train_metrics)
    for i in range(final_epoch):
        metrics.append({
            'epoch': i + 1,
            'train_loss': train_metrics[i]['train_loss'],
            'train_auc': train_metrics[i]['train_auc'],
            'valid_loss': valid_metrics[i]['valid_loss'],
            'valid_auc': valid_metrics[i]['valid_auc'],
            'test_loss': test_metrics[0]['test_loss'], # Test metrics are recorded once at the end
            'test_auc': test_metrics[0]['test_auc'],
        })

    metrics_df = pd.DataFrame(metrics)
    csv_filename = f"QuantumTCNN_freq{args.freq}_sample{args.n_sample}_lr{args.lr}_kernel{args.kernel_size}_dilation{args.dilation}_performance_{seed}.csv"
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")
    
    return test_loss, test_auc



if __name__ == "__main__":
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(seed=args.seed, device=device, batch_size=32, sampling_freq=args.freq, sample_size=args.n_sample)
    print("Input Dimension:", input_dim)
    if args.freq==80:
        QuantumTCNN_run(seed=args.seed, n_qubits=8, circuit_depth=2, input_dim=input_dim, kernel_size=args.kernel_size, dilation=args.dilation, num_epochs=args.epoch, resume=args.resume)
    elif args.freq==4:
        QuantumTCNN_run(seed=args.seed, n_qubits=8, circuit_depth=2, input_dim=input_dim, kernel_size=7, dilation=2, num_epochs=50, resume=args.resume)



