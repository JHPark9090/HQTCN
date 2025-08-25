# 2024 11 24: Modern QLSTM version

# Datetime
from datetime import datetime
import time

import matplotlib.pyplot as plt
from pandas import DataFrame

import warnings

import pennylane as qml
import numpy as np

# Saving
import pickle
import os
import copy

# sklearn
from sklearn.preprocessing import StandardScaler

# Dataset
from Load_PhysioNet_EEG import load_eeg_ts_revised

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=4)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    # parser.add_argument("--resume", action="store_true", default=False)   # resume=True if typing --resume on terminal
    return parser.parse_args()
args=get_args()


##
### Training routine

# --- MODIFICATION: Added accuracy calculation function ---
def calculate_accuracy(y_pred_logit, y_true):
    """Calculates accuracy for binary classification."""
    y_pred_prob = torch.sigmoid(y_pred_logit)
    y_pred_label = (y_pred_prob > 0.5).float()
    correct = (y_pred_label == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy


# def train_epoch_full(opt, model, X, Y, batch_size):
# 	losses = []

# 	for beg_i in range(0, X.shape[0], batch_size):
# 		X_train_batch = X[beg_i:beg_i + batch_size]
# 		# print(x_batch.shape)
# 		Y_train_batch = Y[beg_i:beg_i + batch_size]

# 		# opt.step(closure)
# 		since_batch = time.time()
# 		opt.zero_grad()
# 		# print("CALCULATING LOSS...")
# 		model_res, _ = model(X_train_batch)
# 		loss = nn.MSELoss()
# 		loss_val = loss(model_res.transpose(0,1)[-1], Y_train_batch) # 2024 11 11: .transpose(0,1)
# 		# print("BACKWARD..")
# 		loss_val.backward()
# 		losses.append(loss_val.data.cpu().numpy())
# 		opt.step()
# 		# print("LOSS IN BATCH: ", loss_val)
# 		# print("FINISHED OPT.")
# 		# print("Batch time: ", time.time() - since_batch)
# 		# print("CALCULATING PREDICTION.")
# 	losses = np.array(losses)
# 	return losses.mean()

# --- MODIFICATION: Updated training function ---
def train_epoch(model, optimizer, loader, loss_fn, device):
    losses = []
    accuracies = []
    model.train()
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        
        # Data is (batch, channels, timesteps), model expects (batch, timesteps, channels)
        X_batch = X_batch.permute(0, 2, 1)

        optimizer.zero_grad()
        
        # Get output from the last time step
        model_res, _ = model(X_batch)
        output = model_res[:, -1, :].squeeze(-1)

        # Calculate loss and accuracy
        loss = loss_fn(output, Y_batch)
        acc = calculate_accuracy(output, Y_batch)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
    return np.mean(losses), np.mean(accuracies)

# --- MODIFICATION: Added testing function ---
def test_epoch(model, loader, loss_fn, device):
    losses = []
    accuracies = []
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            X_batch = X_batch.permute(0, 2, 1)

            model_res, _ = model(X_batch)
            output = model_res[:, -1, :].squeeze(-1)

            loss = loss_fn(output, Y_batch)
            acc = calculate_accuracy(output, Y_batch)

            losses.append(loss.item())
            accuracies.append(acc)

    return np.mean(losses), np.mean(accuracies)

##############

### Plotting and Saving

# def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
# 	# Generate file name
# 	file_name = exp_name + "_NO_" + str(exp_index) + "_Epoch_" + str(iteration_list[-1])
# 	saved_simulation_truth = {
# 	"simulation_result" : simulation_result,
# 	"ground_truth" : ground_truth
# 	}

# 	if not os.path.exists(exp_name):
# 		os.makedirs(exp_name)

# 	# Save the train loss list
# 	with open(exp_name + "/" + file_name + "_TRAINING_LOST" + ".txt", "wb") as fp:
# 		pickle.dump(train_loss_list, fp)

# 	# Save the test loss list
# 	with open(exp_name + "/" + file_name + "_TESTING_LOST" + ".txt", "wb") as fp:
# 		pickle.dump(test_loss_list, fp)

# 	# Save the simulation result
# 	with open(exp_name + "/" + file_name + "_SIMULATION_RESULT" + ".txt", "wb") as fp:
# 		pickle.dump(saved_simulation_truth, fp)

# 	# Save the model parameters
# 	torch.save(model.state_dict(), exp_name + "/" +  file_name + "_torch_model.pth")

# 	# Plot
# 	plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list)
# 	plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth)

# 	return


# def plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list):
# 	# Plot train and test loss
# 	fig, ax = plt.subplots()
# 	# plt.yscale('log')
# 	ax.plot(iteration_list, train_loss_list, '-b', label='Training Loss')
# 	ax.plot(iteration_list, test_loss_list, '-r', label='Testing Loss')
# 	leg = ax.legend();

# 	ax.set(xlabel='Epoch', 
# 		   title=exp_name)
# 	fig.savefig(exp_name + "/" + file_name + "_" + "loss" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
# 	plt.clf()

# 	return

# # --- MODIFICATION: Updated saving and plotting functions ---
# def saving(exp_name, exp_index, epoch_list, train_loss_list, val_loss_list, test_loss_list, train_acc_list, val_acc_list, test_acc_list, model):
#     file_name = f"{exp_name}_NO_{exp_index}_Epoch_{epoch_list[-1]}"
    
#     if not os.path.exists(exp_name):
#         os.makedirs(exp_name)

#     # Save losses
#     with open(os.path.join(exp_name, f"{file_name}_TRAINING_LOSS.pkl"), "wb") as fp:
#         pickle.dump(train_loss_list, fp)
#     with open(os.path.join(exp_name, f"{file_name}_VALIDATION_LOSS.pkl"), "wb") as fp: 
#         pickle.dump(val_loss_list, fp)
#     with open(os.path.join(exp_name, f"{file_name}_TESTING_LOSS.pkl"), "wb") as fp:
#         pickle.dump(test_loss_list, fp)
    
#     # Save accuracies
#     with open(os.path.join(exp_name, f"{file_name}_TRAINING_ACC.pkl"), "wb") as fp:
#         pickle.dump(train_acc_list, fp)
#     with open(os.path.join(exp_name, f"{file_name}_VALIDATION_ACC.pkl"), "wb") as fp: 
#         pickle.dump(val_acc_list, fp)
#     with open(os.path.join(exp_name, f"{file_name}_TESTING_ACC.pkl"), "wb") as fp:
#         pickle.dump(test_acc_list, fp)

#     # Save model
#     torch.save(model.state_dict(), os.path.join(exp_name, f"{file_name}_torch_model.pth"))

#     # Plotting
#     plotting_data(exp_name, file_name, epoch_list, train_loss_list, val_loss_list, test_loss_list, "Loss")
#     plotting_data(exp_name, file_name, epoch_list, train_acc_list, val_acc_list, test_acc_list, "Accuracy")

## --- MODIFICATION: This function now saves a full training checkpoint. --- ##
def save_checkpoint(exp_name, epoch, model, optimizer, train_loss_list, val_loss_list, test_loss_list, train_acc_list, val_acc_list, test_acc_list):
    """Saves a training checkpoint and plots the current progress."""
    
    checkpoint_dir = exp_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    ## --- MODIFICATION: The state dictionary now includes the optimizer and epoch number. --- ##
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch_list': list(range(epoch + 1)),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'test_loss_list': test_loss_list,
        'train_acc_list': train_acc_list,
        'val_acc_list': val_acc_list,
        'test_acc_list': test_acc_list,
    }
    
    # This single file contains all information needed to resume training.
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Plotting is called after saving the checkpoint.
    file_name_prefix = f"{exp_name}_Epoch_{epoch+1}"
    plotting_data(checkpoint_dir, file_name_prefix, state['epoch_list'], state['train_loss_list'], state['val_loss_list'], state['test_loss_list'], "Loss")
    plotting_data(checkpoint_dir, file_name_prefix, state['epoch_list'], state['train_acc_list'], state['val_acc_list'], state['test_acc_list'], "Accuracy")


# def plotting_data(exp_name, file_name, epoch_list, train_list, val_list, test_list, metric_name):
#     fig, ax = plt.subplots()
#     ax.plot(epoch_list, train_list, '-b', label=f'Training {metric_name}')
#     ax.plot(epoch_list, val_list, '-g', label=f'Validation {metric_name}')
#     ax.plot(epoch_list, test_list, '-r', label=f'Testing {metric_name}')
#     ax.legend()
#     ax.set(xlabel='Epoch', ylabel=metric_name, title=f'{exp_name} - {metric_name} vs. Epochs')
#     fig.savefig(os.path.join(exp_name, f"{file_name}_{metric_name.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"))
#     plt.close(fig)

## --- MODIFICATION: Plotting now handles np.nan for test metrics that are not yet calculated. --- ##
def plotting_data(exp_dir, file_name_prefix, epoch_list, train_list, val_list, test_list, metric_name):
    fig, ax = plt.subplots()
    ax.plot(epoch_list, train_list, '-b', label=f'Training {metric_name}')
    ax.plot(epoch_list, val_list, '-g', label=f'Validation {metric_name}')
    # Matplotlib automatically skips np.nan values, correctly plotting the test line only at the end.
    ax.plot(epoch_list, test_list, 'r--', label=f'Testing {metric_name}')
    ax.legend()
    ax.set(xlabel='Epoch', ylabel=metric_name, title=f'{metric_name} vs. Epochs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(exp_dir, f"{file_name_prefix}_{metric_name.lower()}_{timestamp}.pdf")
    fig.savefig(plot_path)
    plt.close(fig)

def save_final_metrics_csv(exp_name, epoch_list, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc):
    """Saves the final training, validation, and test metrics to a single CSV file."""
    
    # The directory should already exist from checkpointing, but check just in case.
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    # Create a dictionary of all the metric lists
    metrics_data = {
        'epoch': epoch_list,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'validation_loss': val_loss,
        'validation_accuracy': val_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }

    # Create a pandas DataFrame
    df_metrics = DataFrame(metrics_data)

    # Save the DataFrame to a CSV file
    file_path = os.path.join(exp_name, "final_training_metrics.csv")
    df_metrics.to_csv(file_path, index=False)
    print(f"Final metrics saved to {file_path}")

# def plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth):
# 	# Plot the simulation
# 	plt.axvline(x=train_len, c='r', linestyle='--')
# 	plt.plot(simulation_result, '-')
# 	plt.plot(ground_truth.detach().numpy(), '--')
# 	plt.suptitle(exp_name)
# 	# savfig can only be placed BEFORE show()
# 	plt.savefig(exp_name + "/" + file_name + "_" + "simulation" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
# 	return


#################

## VQC components

##

def H_layer(nqubits):
		"""Layer of single-qubit Hadamard gates.
		"""
		for idx in range(nqubits):
			qml.Hadamard(wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y_tilde axis."""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def entangling_layer(nqubits):
	""" Layer of CNOTs followed by another shifted layer of CNOT."""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])


# Define actual circuit architecture
def q_function(x, q_weights, n_class):
	""" The variational quantum circuit. """

	# Reshape weights
	# θ = θ.reshape(vqc_depth, n_qubits)

	# Start from state |+> , unbiased w.r.t. |0> and |1>

	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	# Embed features in the quantum node
	RY_layer(x)

	# Sequence of trainable variational layers
	# Figure 4: dashed box
	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])

	# Expectation values in the Z basis
	exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]  # only measure first "n_class" of qubits and discard the rest
	return exp_vals


# Wrapped previous model as a PyTorch Module
class VQC(nn.Module):
	def __init__(self, vqc_depth, n_qubits, n_class):
		super().__init__()
		self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # g rotation params
		self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
		self.VQC = qml.QNode(q_function, self.dev, interface = "torch")

		self.n_class = n_class


	def forward(self, X):
		y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)) for x in X]) # PennyLane 0.35.1
		return y_preds.float()

##
##
##

class CustomLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(CustomLSTMCell, self).__init__()
		self.hidden_size = hidden_size

		# Linear layers for gates and cell update
		self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.output_post_processing = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden):
		h_prev, c_prev = hidden

		# Concatenate input and hidden state
		combined = torch.cat((x, h_prev), dim=1)

		# Compute gates
		i_t = torch.sigmoid(self.input_gate(combined))  # Input gate
		f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
		g_t = torch.tanh(self.cell_gate(combined))      # Cell gate
		o_t = torch.sigmoid(self.output_gate(combined)) # Output gate

		# Update cell state
		c_t = f_t * c_prev + i_t * g_t

		# Update hidden state
		h_t = o_t * torch.tanh(c_t)

		# Actual outputs
		out = self.output_post_processing(h_t)

		return out, h_t, c_t

##

class CustomQLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, vqc_depth):
		super(CustomQLSTMCell, self).__init__()
		self.hidden_size = hidden_size

		# Linear layers for gates and cell update
		# Change here to use PEennyLane Quantum VQCs.
		self.input_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.forget_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.cell_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.output_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)

		self.output_post_processing = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden):
		h_prev, c_prev = hidden

		# Concatenate input and hidden state
		combined = torch.cat((x, h_prev), dim=1)

		# Compute gates
		i_t = torch.sigmoid(self.input_gate(combined))  # Input gate
		f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
		g_t = torch.tanh(self.cell_gate(combined))      # Cell gate
		o_t = torch.sigmoid(self.output_gate(combined)) # Output gate

		# Update cell state
		c_t = f_t * c_prev + i_t * g_t

		# Update hidden state
		h_t = o_t * torch.tanh(c_t)

		# Actual outputs
		out = self.output_post_processing(h_t)

		return out, h_t, c_t


##

# class CustomLSTM(nn.Module):
# 	def __init__(self, input_size, hidden_size, lstm_cell_QT):
# 		super(CustomLSTM, self).__init__()
# 		self.hidden_size = hidden_size

# 		# Single LSTM cell
# 		self.cell = lstm_cell_QT

# 	def forward(self, x, hidden=None):
# 		batch_size, seq_len, _ = x.size()

# 		# Initialize hidden and cell states if not provided
# 		if hidden is None:
# 			h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
# 			c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
# 		else:
# 			h_t, c_t = hidden

# 		outputs = []

# 		# Process sequence one time step at a time
# 		for t in range(seq_len):
# 			x_t = x[:, t, :]  # Extract the t-th time step
# 			# print("x_t.shape: {}".format(x_t.shape))
# 			out, h_t, c_t = self.cell(x_t, (h_t, c_t))  # Update hidden and cell states
# 			# print("out: {}".format(out))
# 			outputs.append(out.unsqueeze(1))  # Collect output for this time step

# 		outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
# 		# print("outputs: {}".format(outputs))
# 		return outputs, (h_t, c_t)

# --- MODIFICATION: CustomLSTM now includes a classical embedding layer ---
class CustomLSTM(nn.Module):
    def __init__(self, raw_input_size, feature_embed_size, hidden_size, lstm_cell):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Classical layer to reduce feature dimension
        self.classical_embedding = nn.Linear(raw_input_size, feature_embed_size)
        self.cell = lstm_cell

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_t, c_t = hidden
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            # Apply classical embedding before passing to the cell
            x_embedded = self.classical_embedding(x_t)
            out, h_t, c_t = self.cell(x_embedded, (h_t, c_t))
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h_t, c_t)


# def main():

# 	torch.manual_seed(0)

# 	#

# 	dtype = torch.DoubleTensor

# 	x, y = get_damped_shm_data()

# 	num_for_train_set = int(0.67 * len(x))

# 	x_train = x[:num_for_train_set].type(dtype)
# 	y_train = y[:num_for_train_set].type(dtype)

# 	x_test = x[num_for_train_set:].type(dtype)
# 	y_test = y[num_for_train_set:].type(dtype)

# 	print("x_train: ", x_train)
# 	print("x_test: ", x_test)
# 	print("x_train.shape: ", x_train.shape)
# 	print("x_test.shape: ", x_test.shape)

# 	x_train_transformed = x_train.unsqueeze(2)
# 	x_test_transformed = x_test.unsqueeze(2)

# 	print("x_train: ", x_train_transformed)
# 	print("x_test: ", x_test_transformed)
# 	print("x_train.shape: ", x_train_transformed.shape)
# 	print("x_test.shape: ", x_test_transformed.shape)

# 	print(x_train[0])
# 	print(x_train_transformed[0])

# 	print("y.shape: {}".format(y.shape))


# 	# Example usage
# 	input_size = 1
# 	hidden_size = 5
# 	seq_length = 4
# 	batch_size = 10

# 	output_size = 1

# 	qnn_depth = 5
# 	qlstm_cell = CustomQLSTMCell(input_size, hidden_size, output_size, qnn_depth).double()
	

# 	model = CustomLSTM(input_size, hidden_size, qlstm_cell).double()
	
# 	input_data = torch.randn(batch_size, seq_length, input_size).double()

# 	# Forward pass
# 	output, (h_n, c_n) = model(input_data)

# 	print("Output shape:", output.shape)  # [batch_size, seq_length, hidden_size]
# 	print("Hidden state shape:", h_n.shape)  # [batch_size, hidden_size]
# 	print("Cell state shape:", c_n.shape)  # [batch_size, hidden_size]

# 	print("Output BEFORE transpose: {}".format(output))

# 	output = output.transpose(0,1)
# 	print("Output shape:", output.shape)
# 	print("Output AFTER transpose: {}".format(output))

# 	print(output[-1])

# 	# Check the trainable parameters
# 	print("Show the parameters in QLSTM.")
# 	for name, param in model.named_parameters():
# 		if param.requires_grad:
# 			print(f"Parameter name: {name}")
# 			print(f"Parameter shape: {param.shape}")
# 			# print(f"Parameter grad: {param.grad}")
# 			# print(f"Parameter value: {param.data}\n")

# 	##

# 	exp_name = "QLSTM_TS_MODEL_DAMPED_SHM_1"
# 	exp_index = 1
# 	train_len = len(x_train_transformed)


# 	opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	
# 	train_loss_for_all_epoch = []
# 	test_loss_for_all_epoch = []
# 	iteration_list = []

# 	for i in range(100):
# 		iteration_list.append(i + 1)
# 		train_loss_epoch = train_epoch_full(opt = opt, model = model, X = x_train_transformed, Y = y_train, batch_size = 10)


# 		# Calculate test loss
# 		test_loss = nn.MSELoss()
# 		model_res_test, _ = model(x_test_transformed)
# 		test_loss_val = test_loss(model_res_test.transpose(0,1)[-1], y_test).detach().numpy() # 2024 11 11: .transpose(0,1)
# 		print("TEST LOSS at {}-th epoch: {}".format(i, test_loss_val))

# 		train_loss_for_all_epoch.append(train_loss_epoch)
# 		test_loss_for_all_epoch.append(test_loss_val)

# 		# Run the test
# 		test_run_res, _ = model(x.type(dtype).unsqueeze(2))
# 		total_res = test_run_res.transpose(0,1)[-1].detach().cpu().numpy() # 2024 11 11: .transpose(0,1)
# 		ground_truth_y = y.clone().detach().cpu()

# 		saving(
# 				exp_name = exp_name, 
# 				exp_index = exp_index, 
# 				train_len = train_len, 
# 				iteration_list = iteration_list, 
# 				train_loss_list = train_loss_for_all_epoch, 
# 				test_loss_list = test_loss_for_all_epoch, 
# 				model = model, 
# 				simulation_result = total_res, 
# 				ground_truth = ground_truth_y)

# 	return


# if __name__ == '__main__':
# 	main()


def main():
    # --- MODIFICATION: Updated Hyperparameters and Setup ---
    # Experiment setup
    exp_name = f"QLSTM_EEG_Classification_{args.seed}"
    exp_index = 1
    
    # Model Hyperparameters
    FEATURE_EMBED_SIZE = 4    # Size of the classical embedding
    HIDDEN_SIZE = 4           # LSTM hidden state size
    OUTPUT_SIZE = 1           # Final output is 1 logit for binary classification
    VQC_DEPTH = 5             # Depth of the VQC
    
    # Qubit requirement for VQC gates
    # This is now FEATURE_EMBED_SIZE + HIDDEN_SIZE, which is much more manageable
    N_QUBITS_FOR_GATES = FEATURE_EMBED_SIZE + HIDDEN_SIZE
    
    # Training Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Dataset parameters
    # SAMPLING_FREQ = 4 # Lower sampling freq to reduce sequence length
    # N_SUBJECTS = 50 # Use a smaller subset for faster runs, increase for full experiment

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed, 
        device=device, 
        batch_size=BATCH_SIZE, 
        sampling_freq=args.freq, 
        sample_size=args.n_sample
    )

    raw_input_size = input_dim[1] # Number of EEG channels

    # Instantiate Model
    qlstm_cell = CustomQLSTMCell(FEATURE_EMBED_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, VQC_DEPTH)
    model = CustomLSTM(raw_input_size, FEATURE_EMBED_SIZE, HIDDEN_SIZE, qlstm_cell).to(device)

    # Optimizer and Loss Function
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

#     # --- MODIFICATION: Updated Main Training Loop ---
#     epoch_list = []
#     train_loss_list, val_loss_list, test_loss_list = [], [], []
#     train_acc_list, val_acc_list, test_acc_list = [], [], []

#     print("\nStarting training...")
#     for epoch in range(EPOCHS):
#         start_time = time.time()
        
#         # Training
#         train_loss, train_acc = train_epoch(model, optimizer, train_loader, loss_fn, device)
        
#         # Validation
#         val_loss, val_acc = test_epoch(model, val_loader, loss_fn, device)

#         # Testing
#         test_loss, test_acc = test_epoch(model, test_loader, loss_fn, device)
        
#         epoch_list.append(epoch + 1)
#         train_loss_list.append(train_loss); val_loss_list.append(val_loss); test_loss_list.append(test_loss)
#         train_acc_list.append(train_acc); val_acc_list.append(val_acc); test_acc_list.append(test_acc)
        
#         epoch_time = time.time() - start_time
#         print(f"Epoch {epoch+1}/{EPOCHS} [{epoch_time:.2f}s] - \n"
#               f"\tTrain: Loss={train_loss:.4f}, Acc={train_acc:.4f}\n"
#               f"\tValid: Loss={val_loss:.4f}, Acc={val_acc:.4f}\n"
#               f"\tTest:  Loss={test_loss:.4f}, Acc={test_acc:.4f}")

#         # Save final results
#         saving(
#             exp_name=exp_name,
#             exp_index=exp_index,
#             epoch_list=epoch_list,
#             train_loss_list=train_loss_list,
#             val_loss_list=val_loss_list,
#             test_loss_list=test_loss_list,
#             train_acc_list=train_acc_list,
#             val_acc_list=val_acc_list,  
#             test_acc_list=test_acc_list,
#             model=model
#         )
#         print("Results saved.")

# if __name__ == '__main__':
#     main()


    ## --- MODIFICATION: CHECKPOINT LOADING --- ##
    # This block checks if a checkpoint exists and loads it to resume training.
    start_epoch = 0
    epoch_list = []
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    
    checkpoint_path = os.path.join(exp_name, f"checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_list = checkpoint['epoch_list'] 
        # Load all metric histories to continue logging correctly
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        test_loss_list = checkpoint['test_loss_list']
        train_acc_list = checkpoint['train_acc_list']
        val_acc_list = checkpoint['val_acc_list']
        test_acc_list = checkpoint['test_acc_list']
    else:
        print("No checkpoint found, starting new training run.")

    # --- Main Training Loop ---
    ## --- MODIFICATION: The loop now starts from 'start_epoch'. --- ##
    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, loss_fn, device)
        val_loss, val_acc = test_epoch(model, val_loader, loss_fn, device)

        ## --- MODIFICATION: CONDITIONAL TEST EVALUATION --- ##
        # The test set is now only evaluated on the final epoch to save time.
        if epoch == EPOCHS - 1:
            test_loss, test_acc = test_epoch(model, test_loader, loss_fn, device)
        else:
            # For other epochs, np.nan is used as a placeholder.
            test_loss, test_acc = np.nan, np.nan
        
        # Append results for the current epoch
        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} [{epoch_time:.2f}s] - \n"
              f"\tTrain: Loss={train_loss:.4f}, Acc={train_acc:.4f}\n"
              f"\tValid: Loss={val_loss:.4f}, Acc={val_acc:.4f}\n"
              f"\tTest:  Loss={test_loss:.4f}, Acc={test_acc:.4f}")

        ## --- MODIFICATION: SAVE CHECKPOINT AFTER EACH EPOCH --- ##
        # This call saves the complete state after every epoch, allowing the run to be resumed.
        save_checkpoint(
            exp_name=exp_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            test_loss_list=test_loss_list,
            train_acc_list=train_acc_list,
            val_acc_list=val_acc_list,
            test_acc_list=test_acc_list,
        )

    print("\n--- Training Complete ---")

    ## --- FINAL STEP: SAVE METRICS TO CSV --- ##
    # This call saves all the collected loss and accuracy data to a single CSV file.
    print("Saving final metrics to CSV...")
    save_final_metrics_csv(
        exp_name=exp_name,
        epoch_list=epoch_list,
        train_loss=train_loss_list,
        train_acc=train_acc_list,
        val_loss=val_loss_list,
        val_acc=val_acc_list,
        test_loss=test_loss_list,
        test_acc=test_acc_list
    )

if __name__ == '__main__':
    main()

