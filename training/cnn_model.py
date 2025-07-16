import torch.nn as nn

class CNNModel(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) for feature extraction and classification.

    The model takes a 1D input with shape (batch_size, 1, 240) and performs:
    - Two convolutional blocks with SELU activations, batch normalization, pooling, and dropout.
    - A fully connected block that outputs predictions with a specified number of output units.

    Args:
        output_layer (int): Number of output neurons in the final dense layer,
                            typically matching the number of regression or classification targets.

    Example:
        model = CNNModel(output_layer=100)
    """
    def __init__(self, output_layer):
        super(CNNModel, self).__init__()

        # First convolutional block: Conv1D -> BatchNorm -> SELU -> MaxPool -> Dropout
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),  # Output shape: (N, 64, 238)
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),  # Output shape: (N, 64, 119)
            nn.Dropout(0.2)
        )

        # Second convolutional block: further extract features
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # Output shape: (N, 128, 117)
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),  # Output shape: (N, 128, 58)
            nn.Dropout(0.2)
        )

        # Fully connected block for final predictions
        self.fc_block = nn.Sequential(
            nn.Flatten(),  # Flatten (N, 128, 58) to (N, 128*58 = 7424)
            nn.Linear(128 * 58, 3 * 256),  # Intermediate dense layer
            nn.BatchNorm1d(3 * 256),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(3 * 256, output_layer)  # Final output layer
        )

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x (Tensor): Input tensor of shape (N, 1, 240)

        Returns:
            Tensor: Output predictions of shape (N, output_layer)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x


class FineTunedModel(nn.Module):
    """
    A fine-tuned model that wraps a base CNN model and adds additional dense layers.

    Useful for transfer learning or domain adaptation. Assumes the base model
    outputs features of size 300.

    Args:
        base_model (nn.Module): A pretrained model whose output will be passed
                                through additional trainable layers.

    Example:
        base = CNNModel(output_layer=300)
        model = FineTunedModel(base)
    """
    def __init__(self, base_model):
        super(FineTunedModel, self).__init__()
        self.base_model = base_model  # Pretrained feature extractor (e.g., CNNModel)

        # New fully connected layers for task-specific fine-tuning
        self.new_layers = nn.Sequential(
            nn.Linear(300, 256),
            nn.SELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.SELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 300)  # Adjust this size based on target task
        )

    def forward(self, x):
        """
        Forward pass through the fine-tuned model.

        Args:
            x (Tensor): Input tensor of shape (N, 1, 240)

        Returns:
            Tensor: Output tensor after fine-tuning layers (N, 300)
        """
        x = self.base_model(x)  # Extract features from base model
        x = self.new_layers(x)  # Task-specific transformation
        return x
