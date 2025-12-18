#!/usr/bin/env python3
"""
HQTCN Ablation Study Script

This script runs comprehensive ablation experiments for the HQTCN model:
1. Dilation factor ablation (d = 1, 2, 3, 4)
2. Circuit depth ablation (L = 1, 2, 3)
3. Qubit count ablation (n = 4, 6, 8)
4. Classical MLP baseline comparison (parameter-matched)
5. Noise robustness analysis (depolarizing noise p = 0, 0.001, 0.005, 0.01)
6. Execution time measurements

Author: Junghoon Justin Park
Date: 2025
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Fix scipy lazy loading issue with PennyLane
import scipy.constants  # Must be imported before pennylane
import pennylane as qml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Load_PhysioNet_EEG import load_eeg_ts_revised


# =============================================================================
# Utility Functions
# =============================================================================

def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# =============================================================================
# HQTCN Model (Flexible for Ablation)
# =============================================================================

class HQTCN(nn.Module):
    """
    Hybrid Quantum Temporal Convolutional Network.

    Args:
        n_qubits: Number of qubits in the quantum circuit
        circuit_depth: Number of conv-pool layers
        input_dim: Input dimensions (batch, channels, timesteps)
        kernel_size: Temporal kernel size
        dilation: Dilation factor for temporal windowing
        embedding_dim: Classical embedding dimension (if None, defaults to n_qubits)
        use_noise: Whether to use noisy simulation
        noise_strength: Noise probability (interpretation depends on noise_type)
        noise_type: Type of noise model ('depolarizing', 'amplitude_damping', 'combined')
    """

    def __init__(
        self,
        n_qubits: int,
        circuit_depth: int,
        input_dim: Tuple[int, int, int],
        kernel_size: int,
        dilation: int = 1,
        embedding_dim: Optional[int] = None,
        use_noise: bool = False,
        noise_strength: float = 0.0,
        noise_type: str = 'depolarizing'
    ):
        super(HQTCN, self).__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_noise = use_noise
        self.noise_strength = noise_strength
        self.noise_type = noise_type

        # Embedding dimension (defaults to n_qubits if not specified)
        self.embedding_dim = embedding_dim if embedding_dim is not None else n_qubits

        # Quantum parameters (unscaled, matching original HQTCN2_EEG.py)
        self.conv_params = nn.Parameter(torch.randn(circuit_depth, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(circuit_depth, n_qubits // 2, 3))

        # Classical embedding layers
        # First layer: input -> embedding_dim
        self.fc = nn.Linear(self.input_channels * self.kernel_size, self.embedding_dim)

        # Optional projection layer: embedding_dim -> n_qubits (if they differ)
        if self.embedding_dim != n_qubits:
            self.projection = nn.Linear(self.embedding_dim, n_qubits)
        else:
            self.projection = None

        # Initialize quantum device and circuit
        self._init_quantum_circuit()

    def _init_quantum_circuit(self):
        """Initialize the quantum device and QNode."""
        if self.use_noise and self.noise_strength > 0:
            self.dev = qml.device("default.mixed", wires=self.n_qubits)
            if self.noise_type == 'depolarizing':
                self.quantum_circuit = qml.QNode(self._noisy_circuit_depolarizing, self.dev)
            elif self.noise_type == 'amplitude_damping':
                self.quantum_circuit = qml.QNode(self._noisy_circuit_amplitude_damping, self.dev)
            elif self.noise_type == 'combined':
                self.quantum_circuit = qml.QNode(self._noisy_circuit_combined, self.dev)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.quantum_circuit = qml.QNode(self._circuit, self.dev)

    def _circuit(self, features):
        """Noiseless quantum circuit (matching original HQTCN2_EEG.py)."""
        wires = list(range(self.n_qubits))

        # Angle embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.circuit_depth):
            # Convolutional layer
            self._apply_convolution(self.conv_params[layer], wires)
            # Pooling layer
            self._apply_pooling(self.pool_params[layer], wires)
            # Retain every second qubit after pooling
            wires = wires[::2]

        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _noisy_circuit_depolarizing(self, features):
        """Noisy quantum circuit with depolarizing channel (matching original wire logic)."""
        wires = list(range(self.n_qubits))

        # Angle embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        # Add noise after embedding
        for w in wires:
            qml.DepolarizingChannel(self.noise_strength, wires=w)

        for layer in range(self.circuit_depth):
            # Convolutional layer with noise
            self._apply_convolution_noisy_depolarizing(self.conv_params[layer], wires)
            # Pooling layer (simplified for noisy simulation - no mid-circuit measurement)
            # Retain every second qubit after pooling
            wires = wires[::2]

        return qml.expval(qml.PauliZ(0))

    def _noisy_circuit_amplitude_damping(self, features):
        """Noisy quantum circuit with amplitude damping (T1 decay, matching original wire logic)."""
        wires = list(range(self.n_qubits))

        # Angle embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        # Add amplitude damping after embedding
        for w in wires:
            qml.AmplitudeDamping(self.noise_strength, wires=w)

        for layer in range(self.circuit_depth):
            # Convolutional layer with amplitude damping noise
            self._apply_convolution_noisy_amplitude_damping(self.conv_params[layer], wires)
            # Pooling layer (simplified)
            # Retain every second qubit after pooling
            wires = wires[::2]

        return qml.expval(qml.PauliZ(0))

    def _noisy_circuit_combined(self, features):
        """Noisy quantum circuit with combined depolarizing + amplitude damping (matching original wire logic)."""
        wires = list(range(self.n_qubits))

        # Angle embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        # Add combined noise after embedding
        # Use half the noise strength for each channel to keep total noise comparable
        for w in wires:
            qml.DepolarizingChannel(self.noise_strength / 2, wires=w)
            qml.AmplitudeDamping(self.noise_strength / 2, wires=w)

        for layer in range(self.circuit_depth):
            # Convolutional layer with combined noise
            self._apply_convolution_noisy_combined(self.conv_params[layer], wires)
            # Pooling layer (simplified)
            # Retain every second qubit after pooling
            wires = wires[::2]

        return qml.expval(qml.PauliZ(0))

    def _apply_convolution(self, weights, wires):
        """Apply convolutional layer with U3 and Ising gates."""
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

    def _apply_convolution_noisy_depolarizing(self, weights, wires):
        """Apply convolutional layer with depolarizing noise after each gate."""
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.DepolarizingChannel(self.noise_strength, wires=w)

                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.DepolarizingChannel(self.noise_strength, wires=wires[indx + 1])

                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])

                    qml.DepolarizingChannel(self.noise_strength, wires=w)
                    qml.DepolarizingChannel(self.noise_strength, wires=wires[indx + 1])

                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.DepolarizingChannel(self.noise_strength, wires=w)

                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])
                    qml.DepolarizingChannel(self.noise_strength, wires=wires[indx + 1])

    def _apply_convolution_noisy_amplitude_damping(self, weights, wires):
        """Apply convolutional layer with amplitude damping noise after each gate."""
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.AmplitudeDamping(self.noise_strength, wires=w)

                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.AmplitudeDamping(self.noise_strength, wires=wires[indx + 1])

                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])

                    qml.AmplitudeDamping(self.noise_strength, wires=w)
                    qml.AmplitudeDamping(self.noise_strength, wires=wires[indx + 1])

                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.AmplitudeDamping(self.noise_strength, wires=w)

                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])
                    qml.AmplitudeDamping(self.noise_strength, wires=wires[indx + 1])

    def _apply_convolution_noisy_combined(self, weights, wires):
        """Apply convolutional layer with combined depolarizing + amplitude damping noise."""
        n_wires = len(wires)
        half_noise = self.noise_strength / 2
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.DepolarizingChannel(half_noise, wires=w)
                    qml.AmplitudeDamping(half_noise, wires=w)

                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.DepolarizingChannel(half_noise, wires=wires[indx + 1])
                    qml.AmplitudeDamping(half_noise, wires=wires[indx + 1])

                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])

                    qml.DepolarizingChannel(half_noise, wires=w)
                    qml.AmplitudeDamping(half_noise, wires=w)
                    qml.DepolarizingChannel(half_noise, wires=wires[indx + 1])
                    qml.AmplitudeDamping(half_noise, wires=wires[indx + 1])

                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.DepolarizingChannel(half_noise, wires=w)
                    qml.AmplitudeDamping(half_noise, wires=w)

                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])
                    qml.DepolarizingChannel(half_noise, wires=wires[indx + 1])
                    qml.AmplitudeDamping(half_noise, wires=wires[indx + 1])

    def _apply_pooling(self, pool_weights, wires):
        """Apply pooling layer with mid-circuit measurement."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal windowing."""
        batch_size, input_channels, time_steps = x.size()

        output = []
        start_idx = self.dilation * (self.kernel_size - 1)

        for i in range(start_idx, time_steps):
            # Dilated window indices
            indices = [i - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()

            # Extract and flatten window
            window = x[:, :, indices].reshape(batch_size, -1)

            # Classical embedding
            embedded = self.fc(window)

            # Optional projection to qubit dimension
            if self.projection is not None:
                quantum_input = self.projection(embedded)
            else:
                quantum_input = embedded

            # Quantum circuit
            output.append(self.quantum_circuit(quantum_input))

        # Temporal aggregation
        output = torch.mean(torch.stack(output, dim=1), dim=1)
        return output

    def count_parameters(self) -> Dict[str, int]:
        """Count classical and quantum parameters."""
        classical_params = sum(p.numel() for name, p in self.named_parameters()
                               if 'conv_params' not in name and 'pool_params' not in name)
        quantum_params = self.conv_params.numel() + self.pool_params.numel()
        return {
            'classical': classical_params,
            'quantum': quantum_params,
            'total': classical_params + quantum_params
        }


# =============================================================================
# Classical MLP Baseline (Parameter-Matched)
# =============================================================================

class ClassicalMLP(nn.Module):
    """
    Classical MLP baseline with parameter count matched to HQTCN's quantum parameters.

    This replaces the quantum circuit with a classical MLP of similar capacity.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        kernel_size: int,
        dilation: int,
        hidden_size: int,
        num_hidden_layers: int = 2
    ):
        super(ClassicalMLP, self).__init__()

        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Classical embedding (same as HQTCN)
        self.fc = nn.Linear(self.input_channels * self.kernel_size, hidden_size)

        # MLP layers to replace quantum circuit
        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())  # Bounded activation like quantum output
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())  # Output in [-1, 1] like PauliZ expectation

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal windowing (same as HQTCN)."""
        batch_size, input_channels, time_steps = x.size()

        output = []
        start_idx = self.dilation * (self.kernel_size - 1)

        for i in range(start_idx, time_steps):
            indices = [i - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()

            window = x[:, :, indices].reshape(batch_size, -1)
            reduced_window = self.fc(window)
            mlp_out = self.mlp(reduced_window)
            output.append(mlp_out.squeeze(-1))

        output = torch.mean(torch.stack(output, dim=1), dim=1)
        return output

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())
        return {'classical': total, 'quantum': 0, 'total': total}


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (loss, auroc, epoch_time_seconds)
    """
    model.train()
    train_loss = 0.0
    all_labels = []
    all_outputs = []

    start_time = time.time()

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    epoch_time_sec = time.time() - start_time

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    try:
        train_auroc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        train_auroc = 0.5  # Default if only one class present

    return train_loss / len(dataloader), train_auroc, epoch_time_sec


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluate the model.

    Returns:
        Tuple of (loss, auroc, inference_time_seconds)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    inference_time_sec = time.time() - start_time

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    try:
        auroc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        auroc = 0.5

    return running_loss / len(dataloader), auroc, inference_time_sec


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4
) -> Dict:
    """
    Run a single training experiment.

    Returns:
        Dictionary with all metrics and timing information.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track metrics
    train_losses, train_aurocs = [], []
    val_losses, val_aurocs = [], []
    train_times, val_times = [], []

    best_val_auroc = 0.0
    best_model_state = None

    total_train_time = 0.0

    for epoch in range(num_epochs):
        # Training
        train_loss, train_auroc, train_time = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_aurocs.append(train_auroc)
        train_times.append(train_time)
        total_train_time += train_time

        # Validation
        val_loss, val_auroc, val_time = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_aurocs.append(val_auroc)
        val_times.append(val_time)

        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, AUC: {train_auroc:.4f} | "
              f"Val Loss: {val_loss:.4f}, AUC: {val_auroc:.4f} | Time: {train_time:.1f}s")

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_auroc, test_time = evaluate(
        model, test_loader, criterion, device
    )

    # Calculate average inference time per sample
    n_test_samples = len(test_loader.dataset)
    avg_inference_time = test_time / n_test_samples * 1000  # ms per sample

    return {
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'best_val_auroc': best_val_auroc,
        'final_train_auroc': train_aurocs[-1],
        'total_train_time_sec': total_train_time,
        'avg_epoch_time_sec': total_train_time / num_epochs,
        'avg_inference_time_ms': avg_inference_time,
        'train_losses': train_losses,
        'val_aurocs': val_aurocs,
        'param_counts': model.count_parameters()
    }


# =============================================================================
# Ablation Study Functions
# =============================================================================

def run_dilation_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: Tuple[int, int, int],
    device: torch.device,
    seeds: List[int],
    dilations: List[int] = [1, 2, 3, 4],
    n_qubits: int = 8,
    circuit_depth: int = 2,
    kernel_size: int = 12,
    num_epochs: int = 50
) -> pd.DataFrame:
    """Run ablation study over dilation factors."""

    results = []

    for dilation in dilations:
        print(f"\n{'='*60}")
        print(f"DILATION ABLATION: d = {dilation}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_all_seeds(seed)

            model = HQTCN(
                n_qubits=n_qubits,
                circuit_depth=circuit_depth,
                input_dim=input_dim,
                kernel_size=kernel_size,
                dilation=dilation
            )

            metrics = run_single_experiment(
                model, train_loader, val_loader, test_loader,
                device, num_epochs=num_epochs
            )

            results.append({
                'ablation_type': 'dilation',
                'dilation': dilation,
                'n_qubits': n_qubits,
                'circuit_depth': circuit_depth,
                'kernel_size': kernel_size,
                'seed': seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                **metrics['param_counts']
            })

    return pd.DataFrame(results)


def run_depth_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: Tuple[int, int, int],
    device: torch.device,
    seeds: List[int],
    depths: List[int] = [1, 2, 3],
    n_qubits: int = 8,
    kernel_size: int = 12,
    dilation: int = 3,
    num_epochs: int = 50
) -> pd.DataFrame:
    """Run ablation study over circuit depths."""

    results = []

    for depth in depths:
        print(f"\n{'='*60}")
        print(f"DEPTH ABLATION: L = {depth}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_all_seeds(seed)

            model = HQTCN(
                n_qubits=n_qubits,
                circuit_depth=depth,
                input_dim=input_dim,
                kernel_size=kernel_size,
                dilation=dilation
            )

            metrics = run_single_experiment(
                model, train_loader, val_loader, test_loader,
                device, num_epochs=num_epochs
            )

            results.append({
                'ablation_type': 'depth',
                'dilation': dilation,
                'n_qubits': n_qubits,
                'circuit_depth': depth,
                'kernel_size': kernel_size,
                'seed': seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                **metrics['param_counts']
            })

    return pd.DataFrame(results)


def run_mlp_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: Tuple[int, int, int],
    device: torch.device,
    seeds: List[int],
    kernel_size: int = 12,
    dilation: int = 3,
    num_epochs: int = 50
) -> pd.DataFrame:
    """
    Compare HQTCN with parameter-matched classical MLP.

    The MLP is designed to have approximately the same number of parameters
    as the quantum component of HQTCN.
    """

    results = []

    # HQTCN with 8 qubits, depth 2 has 264 quantum parameters
    # We'll create an MLP with similar capacity
    hidden_size = 8  # Match qubit count

    for model_type in ['HQTCN', 'MLP']:
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON: {model_type}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_all_seeds(seed)

            if model_type == 'HQTCN':
                model = HQTCN(
                    n_qubits=8,
                    circuit_depth=2,
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            else:
                model = ClassicalMLP(
                    input_dim=input_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    hidden_size=hidden_size,
                    num_hidden_layers=2
                )

            metrics = run_single_experiment(
                model, train_loader, val_loader, test_loader,
                device, num_epochs=num_epochs
            )

            results.append({
                'ablation_type': 'model_comparison',
                'model_type': model_type,
                'dilation': dilation,
                'kernel_size': kernel_size,
                'seed': seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                **metrics['param_counts']
            })

    return pd.DataFrame(results)


def run_noise_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: Tuple[int, int, int],
    device: torch.device,
    seeds: List[int],
    n_qubits: int = 8,
    circuit_depth: int = 2,
    kernel_size: int = 12,
    dilation: int = 3,
    num_epochs: int = 50
) -> pd.DataFrame:
    """
    Run noise robustness test with combined noise model.

    Tests noiseless (p=0.0) vs realistic hardware noise (p=0.01).
    Uses combined noise (depolarizing + amplitude damping) to simulate
    realistic near-term quantum hardware conditions.
    """

    results = []

    # Test configurations: noiseless baseline vs realistic combined noise
    noise_configs = [
        {'noise_level': 0.0, 'noise_type': 'none', 'use_noise': False},
        {'noise_level': 0.01, 'noise_type': 'combined', 'use_noise': True},
    ]

    for config in noise_configs:
        noise = config['noise_level']
        noise_type = config['noise_type']
        use_noise = config['use_noise']

        print(f"\n{'='*60}")
        print(f"NOISE TEST: {'Noiseless' if not use_noise else f'Combined noise (p={noise})'}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_all_seeds(seed)

            model = HQTCN(
                n_qubits=n_qubits,
                circuit_depth=circuit_depth,
                input_dim=input_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                use_noise=use_noise,
                noise_strength=noise,
                noise_type='combined' if use_noise else 'depolarizing'
            )

            metrics = run_single_experiment(
                model, train_loader, val_loader, test_loader,
                device, num_epochs=num_epochs
            )

            results.append({
                'ablation_type': 'noise',
                'noise_type': noise_type,
                'noise_level': noise,
                'n_qubits': n_qubits,
                'circuit_depth': circuit_depth,
                'kernel_size': kernel_size,
                'dilation': dilation,
                'seed': seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                **metrics['param_counts']
            })

    return pd.DataFrame(results)


def run_embedding_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: Tuple[int, int, int],
    device: torch.device,
    seeds: List[int],
    embedding_dims: List[int] = [4, 8, 16, 32],
    n_qubits: int = 8,
    circuit_depth: int = 2,
    kernel_size: int = 12,
    dilation: int = 3,
    num_epochs: int = 50
) -> pd.DataFrame:
    """
    Run ablation study over embedding dimensions.

    This isolates the effect of embedding size from qubit count.
    The quantum circuit stays fixed at n_qubits, but the classical
    embedding dimension varies.
    """

    results = []

    for emb_dim in embedding_dims:
        print(f"\n{'='*60}")
        print(f"EMBEDDING ABLATION: embedding_dim = {emb_dim}, n_qubits = {n_qubits}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            set_all_seeds(seed)

            model = HQTCN(
                n_qubits=n_qubits,
                circuit_depth=circuit_depth,
                input_dim=input_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                embedding_dim=emb_dim
            )

            metrics = run_single_experiment(
                model, train_loader, val_loader, test_loader,
                device, num_epochs=num_epochs
            )

            results.append({
                'ablation_type': 'embedding',
                'embedding_dim': emb_dim,
                'n_qubits': n_qubits,
                'circuit_depth': circuit_depth,
                'kernel_size': kernel_size,
                'dilation': dilation,
                'seed': seed,
                **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                **metrics['param_counts']
            })

    return pd.DataFrame(results)


# =============================================================================
# Main Entry Point
# =============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HQTCN Ablation Study')

    # Data parameters
    parser.add_argument('--freq', type=int, default=80,
                        help='Sampling frequency (default: 80)')
    parser.add_argument('--n-sample', type=int, default=50,
                        help='Number of subjects (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')

    # Experiment parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025, 2026, 2027],
                        help='Random seeds (default: 2025 2026 2027)')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')

    # Ablation selection
    parser.add_argument('--ablation', type=str, nargs='+',
                        default=['dilation', 'depth', 'embedding', 'mlp', 'noise'],
                        choices=['dilation', 'depth', 'embedding', 'mlp', 'noise', 'all'],
                        help='Which ablations to run')

    # Single configuration mode (for parallel job arrays)
    parser.add_argument('--single-dilation', type=int, default=None,
                        help='Run single dilation value (for parallel jobs)')
    parser.add_argument('--single-depth', type=int, default=None,
                        help='Run single depth value (for parallel jobs)')

    # Output
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help='Directory to save results')

    return parser.parse_args()


def main():
    """Main entry point for ablation studies."""
    args = get_args()

    # Setup
    device = get_device()
    print(f"Running on device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Ablations: {args.ablation}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load data (using first seed for data split)
    print("\nLoading data...")
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seeds[0],
        device=device,
        batch_size=args.batch_size,
        sampling_freq=args.freq,
        sample_size=args.n_sample
    )
    print(f"Input dimension: {input_dim}")

    # Determine which ablations to run
    if 'all' in args.ablation:
        ablations_to_run = ['dilation', 'depth', 'embedding', 'mlp', 'noise']
    else:
        ablations_to_run = args.ablation

    all_results = []

    # Run ablations
    if 'dilation' in ablations_to_run:
        print("\n" + "="*80)
        print("RUNNING DILATION ABLATION")
        print("="*80)
        # Single configuration mode or full sweep
        if args.single_dilation is not None:
            dilations = [args.single_dilation]
            print(f"Single configuration mode: dilation={args.single_dilation}")
        else:
            dilations = [1, 2, 3, 4]
        df = run_dilation_ablation(
            train_loader, val_loader, test_loader, input_dim,
            device, args.seeds, dilations=dilations, num_epochs=args.num_epochs
        )
        df.to_csv(os.path.join(output_dir, 'dilation_ablation.csv'), index=False)
        all_results.append(df)
        print(f"\nDilation ablation summary:")
        print(df.groupby('dilation')['test_auroc'].agg(['mean', 'std']))

    if 'depth' in ablations_to_run:
        print("\n" + "="*80)
        print("RUNNING DEPTH ABLATION")
        print("="*80)
        # Single configuration mode or full sweep
        if args.single_depth is not None:
            depths = [args.single_depth]
            print(f"Single configuration mode: depth={args.single_depth}")
        else:
            depths = [1, 2, 3]
        df = run_depth_ablation(
            train_loader, val_loader, test_loader, input_dim,
            device, args.seeds, depths=depths, num_epochs=args.num_epochs
        )
        df.to_csv(os.path.join(output_dir, 'depth_ablation.csv'), index=False)
        all_results.append(df)
        print(f"\nDepth ablation summary:")
        print(df.groupby('circuit_depth')['test_auroc'].agg(['mean', 'std']))

    if 'embedding' in ablations_to_run:
        print("\n" + "="*80)
        print("RUNNING EMBEDDING ABLATION")
        print("="*80)
        df = run_embedding_ablation(
            train_loader, val_loader, test_loader, input_dim,
            device, args.seeds, num_epochs=args.num_epochs
        )
        df.to_csv(os.path.join(output_dir, 'embedding_ablation.csv'), index=False)
        all_results.append(df)
        print(f"\nEmbedding ablation summary:")
        print(df.groupby('embedding_dim')['test_auroc'].agg(['mean', 'std']))

    if 'mlp' in ablations_to_run:
        print("\n" + "="*80)
        print("RUNNING MLP COMPARISON")
        print("="*80)
        df = run_mlp_comparison(
            train_loader, val_loader, test_loader, input_dim,
            device, args.seeds, num_epochs=args.num_epochs
        )
        df.to_csv(os.path.join(output_dir, 'mlp_comparison.csv'), index=False)
        all_results.append(df)
        print(f"\nMLP comparison summary:")
        print(df.groupby('model_type')['test_auroc'].agg(['mean', 'std']))

    if 'noise' in ablations_to_run:
        print("\n" + "="*80)
        print("RUNNING NOISE ROBUSTNESS TEST")
        print("="*80)
        df = run_noise_ablation(
            train_loader, val_loader, test_loader, input_dim,
            device, args.seeds, num_epochs=args.num_epochs
        )
        df.to_csv(os.path.join(output_dir, 'noise_robustness.csv'), index=False)
        all_results.append(df)
        print(f"\nNoise robustness summary (noiseless vs combined p=0.01):")
        print(df.groupby('noise_type')['test_auroc'].agg(['mean', 'std']))

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'all_ablations.csv'), index=False)
        print(f"\nAll results saved to {output_dir}")

    # Print final summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    main()
