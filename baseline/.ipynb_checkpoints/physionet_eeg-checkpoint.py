import mne
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

def get_physionet_eeg_data(seed=2025, batch_size=32, sampling_freq=80, n_subjects=50, device='cpu'):
    """
    Loads and preprocesses the PhysioNet EEG Motor Imagery dataset for a specified number of subjects.

    Args:
        seed (int): Random seed for reproducibility.
        batch_size (int): Number of samples per batch.
        sampling_freq (int): The target sampling frequency to resample the data to.
        n_subjects (int): The number of subjects to load data from (1 to 109).
        device (torch.device): The device to move the tensors to.

    Returns:
        tuple: A tuple containing (train_loader, test_loader, input_dim).
               - train_loader: DataLoader for the training set.
               - test_loader: DataLoader for the test set.
               - input_dim: The shape of the input data (n_samples, n_channels, n_timesteps).
    """
    # define the preferred download path
    download_path = os.path.expanduser('~/PhysioNet_EEG_data')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Define the task: imagining opening/closing left or right fist
    # Runs 4, 8, 12 correspond to this task.
    IMAGINE_FIST_RUNS = [4, 8, 12]

    # Load data from PhysioNet for each subject
    all_raw_parts = []
    for subj_id in range(1, n_subjects + 1):
        print(f"Loading and processing data for subject {subj_id}/{n_subjects}...")
        try:
            physionet_paths = mne.datasets.eegbci.load_data(
                subjects=[subj_id],
                runs=IMAGINE_FIST_RUNS,
                path=download_path,
                update_path=True # Set to True to download if not found
            )
            
            for path in physionet_paths:
                raw = mne.io.read_raw_edf(
                    path,
                    preload=True,
                    stim_channel='auto',
                    verbose='WARNING',
                )
                # Resample raw data to ensure consistent sfreq
                raw.resample(sampling_freq, npad="auto")
                all_raw_parts.append(raw)
        except Exception as e:
            print(f"Could not load data for subject {subj_id}. Error: {e}")
            continue
    
    if not all_raw_parts:
        raise RuntimeError("No data could be loaded. Please check your internet connection and file paths.")

    print("\nConcatenating data from all subjects...")
    raw_concatenated = mne.concatenate_raws(all_raw_parts)

    # Pick EEG channels and extract events
    eeg_channel_inds = mne.pick_types(
        raw_concatenated.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    events, _ = mne.events_from_annotations(raw_concatenated, verbose='WARNING')

    # Epoch the data (extracting trials)
    # Event IDs: 2 for left fist, 3 for right fist
    epoched = mne.Epochs(
        raw_concatenated, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True, verbose='WARNING'
    )

    # Convert data to NumPy arrays
    # Shape: (n_epochs, n_channels, n_times)
    X = (epoched.get_data() * 1e6).astype(np.float32)  # Convert to microvolts for better scaling
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # Labels: 0 for left, 1 for right

    # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    
    # Split data into 70% training and 30% temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    # Split the 30% temporary set into 15% validation and 15% test set (50/50 split of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    def make_tensor_dataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device) # BCEWithLogitsLoss expects float
        return TensorDataset(X_tensor, y_tensor)

    train_dataset = make_tensor_dataset(X_train, y_train)
    val_dataset = make_tensor_dataset(X_val, y_val)
    test_dataset = make_tensor_dataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape
    print(f"\nData loading complete. Input data shape: {input_dim}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Testing samples: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, input_dim