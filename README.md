# Wine Quality Prediction with Deep Learning 
The dataset used in this project contains various parameters that influence wine quality.<br>
We are predicting wine quality based on 11 features, which will serve as inputs to our neural network.<br>
The input features are fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulfates, and alcohol. <br>
The neural network output will be the quality of the wine, represented by a number from 4 to 8 in our dataset. <br>
You can find the dataset at the following link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset <br>
<br>

### Model Overview
The model employed is a feed-forward fully connected neural network. <br>
First, the data is read from the CSV file and then converted into tensors. This allows a user to utilize the data directly without needing to split it. <br>
The data is subsequently divided into a training set and a testing set. <br>
The model is then trained using the training set, which consists of labeled data that enables the model to learn. <br>
Finally, the model is evaluated with the testing set. <br>








 The model used is a feed-forward fully connected neural network.
## First of all, the data is read from the CSV file, then converted into tensors in case a user wants to directly use the data without splitting it

then divided into training set and testing set
## Next, the model is trained using the training set, which is labeled data so that the model can learn
## Finally, the model is tested by being given the testing set


# How to use the code
## The code is divided into 6 main functions:
### Data_set(filename, input_features, output_features): Calling this function will extract from your data the input and output data. You need to pass to the function filename, which will be the name of the file that contains the data, input_features: the name of the columns that you want to extract as input, and output features the name of the columns that you want to extract as output to train the model. The main purpose for putting the output of this fucntion as tensors is that if someone wants to use the entire data on the neural network without splitting, they can directly use.

### Data_set_split(input_tensor, output_tensor, test_size=0.2): Calling this function splits the data into training set and testing set. You need to pass to this function the input and output tensors from the Data_set function. test_size is by default set to 0.2, meaning that 20% of the data will be for testing and the rest will be for training, but the user can input another number to change the percentage of splitting data.

### class NeuralNet(nn.Module): This class defines the architecture of the neural network, with the dimensions of the different layers, and activation function on each layer.

### create_model(architecture,activation_function). This function uses the class NeuralNet to create a neural network. To specify the architecture (number of hidden layers and their dimensions) and the activation function, the user can pass as arguments to the function an array architecture that contains the dimensions of the layers in order (starting from the input, until the output). The user can also set an activation function of his choosing by passing it as an argument to the create model function

### train_model(train_input, train_output, optimizer_type, model, epochs, learning_rate): This function is used to train the model. It takes the training input, training output (got from splitting) as arguments. To call this function, you need also to pass other arguments like the optimizer_type, the model (created using the create_model function), number of epochs and the learning rate

### test_loss(model,input_testing,output_testing, criterion): The function is used to get the error on the test data. To use this function, you need to pass your model, the test input and output data. The output of the function is the loss.

