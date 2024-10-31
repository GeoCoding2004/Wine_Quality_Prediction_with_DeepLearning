# Wine Quality Prediction with Deep Learning 
The dataset used in this project contains various parameters that influence wine quality.<br>
We are predicting wine quality based on 11 features, which will serve as inputs to our neural network.<br>
The input features are fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol. <br>
The neural network output will be the quality of the wine, represented by a number from 4 to 8 in our dataset. <br>
You can find the dataset at the following link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset <br>

## Model Overview
The model employed is a feed-forward fully connected neural network. <br>
First, the data is read from the CSV file and then converted into tensors. This allows a user to utilize the data directly without needing to split it. <br>
The data is subsequently divided into a training set and a testing set. <br>
The model is then trained using the training set, which consists of labeled data that enables the model to learn. <br>
Finally, the model is evaluated with the testing set. <br>
<br>

# How to Use the Code 
The code is organized into six main functions: <br>

### 1. Data_set(filename, input_features, output_features) 
This function extracts input and output data from your dataset. You need to provide the filename (the name of the file containing the data), the input_features and output_features (Array containing strings of names of the columns to extract as input or output). The extracted data is then converted into tensors to be used in the neural network. This function returns two tensors of the input and output.
<br> Example: If the user wants to extract from the data the 'fixed acidity' as input the 'quality' as output, the following function shows should be used: <br> input_tensor, output_tensor = Data_set(filename, ['fixed acidity',....], ['quality']) <br>

### 2. Data_set_split(input_tensor, output_tensor, test_size=0.2) 
This function splits the data into a training set and a testing set. You need to pass the input and output tensors obtained from the Data_set function. The default test_size is set to 0.2, meaning that 20% of the data will be used for testing and the remaining 80% for training. However, users can specify a different value to adjust the splitting percentage. <br>

### 3. Class NeuralNet(nn.Module) 
This class defines the architecture of the neural network, including the dimensions of the different layers and the activation function employed in each layer. <br>

### 4. Create_model(architecture, activation_function) 
This function creates a neural network using the NeuralNet class. You can specify the architecture (the number of hidden layers and their dimensions) and the activation function. Simply pass an array representing the layer dimensions (starting from the input layer to the output) and your chosen activation function as arguments to this function. <br>\

### 5. Train_model(train_input, train_output, optimizer_type, model, epochs, learning_rate) 
This function trains the model. It requires the training input and output obtained from the splitting function as arguments. Additionally, you must provide the optimizer type, the model (created using the create_model function), the number of epochs, and the learning rate. <br>

### 6. Test_loss(model, input_testing, output_testing, criterion) 
This function calculates the error on the test data. To use it, you need to pass your model along with the test input and output data. The function will return the calculated loss.



