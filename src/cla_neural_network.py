import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CLANeuralNetwork:
    def __init__(self, input_cols, output_cols, hidden_units=64, learning_rate=0.001):
        """
        Initializes the neural network with the provided DataFrame and parameters.

        :param dataframe: pandas DataFrame containing the data.
        :param input_cols: List of column names to be used as inputs.
        :param output_cols: List of column names to be used as outputs.
        :param hidden_units: Number of neurons in the hidden layer.
        :param learning_rate: Learning rate for the optimizer.
        """
        # self.dataframe = dataframe
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.learning_rate = learning_rate
        self.model = self._build_model(hidden_units)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []

    def _build_model(self, hidden_units):
        input_size = len(self.input_cols)
        output_size = len(self.output_cols)
        model = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size)
        )
        return model

    def train(self, dataframe, epochs=100, batch_size=32, validation_split=0.2):
        """
        Trains the neural network.

        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        :param validation_split: Fraction of data to use for validation.
        """
        # Prepare data
        self.dataframe = dataframe
        data_size = len(self.dataframe)
        val_size = int(data_size * validation_split)
        train_df = self.dataframe[:-val_size]
        val_df = self.dataframe[-val_size:]

        X_train = torch.tensor(train_df[self.input_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_df[self.output_cols].values, dtype=torch.float32)
        # X_val = torch.tensor(val_df[self.input_cols].values, dtype=torch.float32)
        # y_val = torch.tensor(val_df[self.output_cols].values, dtype=torch.float32)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Initialize a variable to store the final epoch's loss.
        final_epoch_loss = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            # Calculate the average loss for this epoch.
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            final_epoch_loss = avg_epoch_loss  # Update final_epoch_loss with the most recent epoch's loss
            # print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')

        # Append the final epoch's loss only once per call to train.
        if final_epoch_loss is not None:
            self.losses.append(final_epoch_loss)
            # print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
            # Optionally, evaluate on validation set
            # if validation_split > 0:
            #     self.model.eval()
            #     with torch.no_grad():
            #         val_outputs = self.model(X_val)
            #         val_loss = self.criterion(val_outputs, y_val)
                # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            # else:
                # print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    def predict(self, input_data):
        """
        Predicts the output for given input data.

        :param input_data: pandas DataFrame or 2D numpy array with the same input columns.
        :return: Predicted values as a numpy array.
        """
        if isinstance(input_data, pd.DataFrame):
            X_new = input_data[self.input_cols].values
        else:
            X_new = np.array(input_data)
        X_new = torch.tensor(X_new, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_new).numpy()
        return predictions

    def evaluate(self, test_dataframe):
        """
        Evaluates the model on a test DataFrame.

        :param test_dataframe: pandas DataFrame containing test data.
        :return: Loss value.
        """
        X_test = torch.tensor(test_dataframe[self.input_cols].values, dtype=torch.float32)
        y_test = torch.tensor(test_dataframe[self.output_cols].values, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test)
        return loss.item()