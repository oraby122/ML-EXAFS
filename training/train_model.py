import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, epochs, learning_rate, train_generator, val_loader, device, steps_per_epoch,weight_decay):
    """
    Train a PyTorch model using data from a generator.

    Args:
        model: The PyTorch model to train.
        epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        train_generator: A generator function that yields (inputs, targets) for training.
        val_loader: DataLoader for validation data.
        device: The device (CPU or GPU) to train on.
        steps_per_epoch: Number of batches per epoch.
        batch_size: Batch size used in training.
    """
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    model.to(device)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Loop through epochs
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        # Iterate over training data
        for step in range(steps_per_epoch):
            # Get a batch from the generator
            x_batch, y_batch = next(train_generator)
            
            # Convert to PyTorch tensors and send to device
            x_batch = torch.tensor(x_batch, dtype=torch.float32).permute(0,2,1).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)

            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_train_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_train_loss / steps_per_epoch
        train_losses.append(avg_train_loss)
        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}")
        
    history = {"train_loss": train_losses, "val_loss": val_losses}
    return model,history
