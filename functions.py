import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split


    # Assuming that the data is in a CSV file
    def Data_set(filename, input_features, output_features):
        input_data = pd.read_csv(filename, usecols= input_features).astype(np.float32)
        output_data = pd.read_csv(filename, usecols=output_features).astype(np.float32)
    
        # Convert data to tensors
        input_tensor = torch.tensor(input_data.values)
        output_tensor = torch.tensor(output_data.values)
        
        return input_tensor, output_tensor


    # Splitting the data into training set and testing set
    def Data_set_split(input_tensor, output_tensor, test_size=0.2):
        #Convert to numpy for compatibility with train_test_split
        input_numpy = input_tensor.numpy()
        output_numpy = output_tensor.numpy()
        
        # Split the data (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(input_numpy, output_numpy, test_size=test_size, random_state=42)
        
        # Convert back to tensors
        input_train = torch.tensor(X_train, requires_grad=True)
        output_train = torch.tensor(y_train, requires_grad=True)
        input_test = torch.tensor(X_test, requires_grad=True)
        output_test = torch.tensor(y_test, requires_grad=True)
        
        return input_train,output_train,input_test,output_test
       
    
   
    # Define the neural network
    class NeuralNet(nn.Module):
        def __init__(self, architecture, activation_function):
            super(NeuralNet, self).__init__()
            self.layers = nn.ModuleList()
            self.activation_function = activation_function  # Store the activation function as an attribute
        
            for i in range(len(architecture)-1):
                self.layers.append(nn.Linear(architecture[i], architecture[i+1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all but last layer
            if i < len(self.layers) - 1:
                x = self.activation_function(x)
        return x
    

    # Create the model used to represent the neural network
    def Create_model(architecture,activation_function):
        # Instantiate the network
        return NeuralNet(architecture,activation_function)

    
    
    def Train_model(train_input, train_output, optimizer_type, model, epochs, learning_rate):
      # Define mean squared error loss
      criterion = nn.MSELoss()
    
      # Define optimizer
      optimizer = optimizer_type(model.parameters(), learning_rate)

    # Training loop
    for epoch in range(epochs):  # epochs
        # Zero the gradients
        optimizer.zero_grad()
    
        # Forward pass
        predicted_output = model(input_training)
        
        # Calculate loss
        loss = criterion(predicted_output, output_training)
        
        # Backpropagation
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {loss.item()}")


    # Finding the test loss of the neural network 
    def Test_loss(model,input_testing,output_testing, criterion):
        test_predicted = model(input_testing)
        test_loss = criterion(test_predicted, output_testing)
        return test_loss.item()


    # Predicing the output of an input using the model
    def Prediction(model, input_testing): 
        with torch.no_grad():
            # We don't need gradients for prediction
            predicted = model(input_testing)
            # Convert tensor to float
            return predicted.item() if predicted.numel() == 1 else predicted.numpy()
