import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, output_layer):
        super(CNNModel,self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),  # Input shape (N, C=1, L=240)
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),  # Flatten for dense layers
            nn.Linear(128 * 53, 3 * 256),  # Input size after pooling
            nn.BatchNorm1d(3 * 256),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(3 * 256, output_layer)  # Output layer
        )
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x
    
class FineTunedModel(nn.Module):
    def __init__(self, base_model):
        super(FineTunedModel, self).__init__()
        self.base_model = base_model  # Keep base model as is
        
        # Additional layers for fine-tuning
        self.new_layers = nn.Sequential(
            nn.Linear(300, 256),  # Assuming base model outputs 115 features
            nn.SELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 300)  # Final output layer with 100 outputs
        )

    def forward(self, x):
        x = self.base_model(x)  # Pass through the frozen base model
        x = self.new_layers(x)  # Pass through the new layers
        return x
