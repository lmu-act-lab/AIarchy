import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def train_model(self, df, input_vars, output_var, epochs=100, learning_rate=0.01):
        # Extract input and output from DataFrame
        X = torch.tensor(df[input_vars].values, dtype=torch.float32)
        y = torch.tensor(df[output_var].values, dtype=torch.float32).unsqueeze(1)

        # Define the loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f'Training complete. Final loss: {loss.item()}')

    def get_prediction(self, input_data):
        # Ensure the model is in evaluation mode
        self.eval()

        # Convert input data to tensor if it's a DataFrame or list
        if isinstance(input_data, pd.DataFrame) or isinstance(input_data, list):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data  # Assumes input is already a tensor

        # Add batch dimension if input is a single instance
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Perform the forward pass to get prediction
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(input_tensor)
        
        return output

import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.rand(100) * 10  # Scale target to make it interesting
}
df = pd.DataFrame(data)

# Initialize the model
input_size = 2  # Number of features (feature1 and feature2)
hidden_size = 10
output_size = 1
model = SimpleNet(input_size, hidden_size, output_size)

# Train the model on the synthetic data
model.train_model(df, input_vars=['feature1', 'feature2'], output_var='target', epochs=200)

# Make a prediction with new input data
new_input = [0.6, 0.2]  # Example input values
pred = get_prediction(model, new_input)
print("Prediction for new input:", pred.item())