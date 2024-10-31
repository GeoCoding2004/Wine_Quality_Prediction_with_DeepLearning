# Wine_Quality_Prediction_with_DeepLearning

## The dataset used showcases different parameters that dictate the quality of the wine.
## We are predicting the wine quality based on 11 features which will be the input to our neural network.
## Those inputs are: fixed acidity,	volatile acidity, citric acid,	residual sugar,	chlorides,	free sulfur dioxide,	total sulfur dioxide,	density,	pH	,sulphates,	alcohol
## On the other side of the neural network is the quality of the wine, which is a number from 1 to 10 (Although our dataset only contains labeled data from 4 up to 8)

### The model used is a feed forward fully connected neural network.
## First of all, the data is read from the CSV file, then divided into training set and testing set
## Next, the model is trained using the training set, which has labeled data so that the model can learn
## Finally, the model is tested by being given the testing set
