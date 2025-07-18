# %%
###########################
# Importing Libraries
###########################
# General libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Custom libraries
from training_tools import data_generator

# Machine learning libraries
import torch
from torchinfo import summary
from cnn_model import CNNModel, FineTunedModel
from train_model import train_model
from torch.utils.data import DataLoader, TensorDataset
#%%
##############################
# Hyperparameters
##############################
epochs = 25
batch_size = 64
weight_decay = 1e-5 # For L2 Regularization penalty
learning_rate = 0.0001
steps_per_epochs = int(40000 / batch_size) # No. of batches per epoch for "data_generator"

# Set for GPU-accelerated training otherwise omit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU
#%%
##############################
# Data Loading and processing
##############################

# Load preprocessed training data (macrostates) from a pickle file.
all_macrostates = pickle.load(open("../data/dataset12/onne_macrostates.pkl", "rb"))

# Filter out any samples where the maximum value of the input exceeds 2
all_macrostates = [example for example in all_macrostates if np.max(example[0]) < 2]

# Convert to numpy object array for compatibility with custom generator
all_macrostates = np.array(all_macrostates,dtype=object)

# Count total number of training samples
num_special_bois = len(all_macrostates)

# Initialize custom data generator for training
train_generator = data_generator(batch_size, num_special_bois,all_macrostates)

# Validation dataset
x_val = pickle.load(open("../data/dataset12/x_val_full_rdf.pkl", "rb")).astype(np.float32)
y_val = pickle.load(open("../data/dataset12/y_val_full_rdf.pkl", "rb")).astype(np.float32)

# Convert validation data to PyTorch tensors
x_val = torch.from_numpy(x_val)
y_val = torch.from_numpy(y_val)

#  Wrap validation tensors into a PyTorch `TensorDataset` for batching
validation_dataset = TensorDataset(x_val,y_val)
# Create a PyTorch DataLoader to serve batches during validation
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
#%%
##############################
# Model - Training
##############################
output_layer = len(y_val[0]) # Output layer is equal to the size of features
model = CNNModel(output_layer) # Instantiate a model
print(summary(model,input_size=(val_dataloader.batch_size, 1, 220))) # Print model architecture
working_dir = "base_model"
#%%

# Ensure the output directory exists (create if not)
os.makedirs(working_dir,exist_ok=True)

# Loop over K-folds (in this case, 10-fold training)
for fold in range(10):
    # Instantiate a new model for each fold
    model = CNNModel(output_layer) 

    # Train the model for this fold
    model, history= train_model(
                model=model,
                epochs=epochs,
                learning_rate=learning_rate,
                train_generator=train_generator,
                val_loader=val_dataloader,
                device=device,
                steps_per_epoch=steps_per_epochs,
                weight_decay=weight_decay
    )
# Save model checkpoint, training history, and hyperparameters
    save_point = {
        "model_state_dict": model.state_dict(),  # Trained model weights
        "epoch": epochs,                         # Total number of training epochs
        "loss": history,                         # Training and validation loss history
        "hyperparameters": {                     # Record of hyperparameters used
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "L2 Regularization": weight_decay
        }
    }
    # Define path for saving the model checkpoint for this fold
    save_path = os.path.join(working_dir, f"base_model_{fold}.pth")

    # Save the checkpoint to disk
    torch.save(save_point, save_path)
# %%
##############################
# Model - Fine Tuning
##############################

# Load base model checkpoints trained during pretraining
base_models = glob(os.path.join(working_dir,"base_model_*"))

# Load fine-tuning training data with same treatment as ONNE Data
NNMD_macrostates = pickle.load(open("../data/dataset12/NNMD_all_traj_all_temps.pkl", "rb"))
NNMD_macrostates = np.asarray([example for example in NNMD_macrostates if np.max(example[0]) < 2],dtype=object)

num_special_bois = len(NNMD_macrostates)
train_generator = data_generator(batch_size, num_special_bois,NNMD_macrostates)

NNMD_x_val = pickle.load(open("../data/dataset12/NNMD_x_val_full_rdf.pkl", "rb")).astype(np.float32)
NNMD_y_val = pickle.load(open("../data/dataset12/NNMD_y_val_full_rdf.pkl", "rb")).astype(np.float32)

NNMD_x_val = torch.from_numpy(NNMD_x_val)
NNMD_y_val = torch.from_numpy(NNMD_y_val)

ft_val_dataset = TensorDataset(NNMD_x_val,NNMD_y_val)
ft_val_dataloader = DataLoader(ft_val_dataset, batch_size=batch_size, shuffle=True)
#%%
# Loop through all base models and fine-tune them
for fold, model in enumerate(base_models):
    # Load base model checkpoint
    checkpoint = torch.load(model)

    # Recreate base model architecture and load pretrained weights
    base_model = CNNModel(output_layer)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.to(device)

    # Freeze all base model parameters to prevent updates during fine-tuning
    for param in base_model.parameters():
        param.requires_grad = False

    # Create a new model that adds trainable layers on top of the frozen base
    fine_tuned_model = FineTunedModel(base_model).to(device)

    # Fine-tune on the new dataset
    ft_model, history = train_model(
        model=fine_tuned_model,
        epochs=epochs,
        learning_rate=learning_rate,
        train_generator=train_generator,
        val_loader=ft_val_dataloader,
        device=device,
        steps_per_epoch=steps_per_epochs,
        weight_decay=weight_decay
    )

    # Save the fine-tuned model checkpoint
    save_point = {
        "model_state_dict": ft_model.state_dict(),
        "epoch": epochs,
        "loss": history,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "L2 Regularization": weight_decay
        }
    }

    # Save
    save_path = os.path.join(working_dir, f"ft_model_{fold}.pth")
    torch.save(save_point, save_path)