# Keras Tutorial Project
This repository is destined to create a tutorial to learn keras. We have used the dataset Poker Hand that is availavle in this link: http://archive.ics.uci.edu 

The tutorial includes the following topics:
  - [Loading Data](#LOADING-DATA)
 - [Creating Model](#CREATING-MODEL)
 - [Compiling Model](#COMPILING-MODEL)
 - [Fitting Model](#FITTING-MODEL)
 - [Evaluating Model](#EVALUATING-MODEL)
 - [Tuning Parameters](#TUNING-MODEL)

## Data Description:
The Poker Hand database consists of 1,025,010 instances of poker hands. Each instance is an example of a poker hand consisting of five cards drawn from a standard deck of 52 cards. Each card is described using two attributes (suit and rank), for a total of 10 features. There is one Class attribute that describes the Poker Hand. The order of cards is important, which is why there are 480 possible Royal Flush hands as compared to 4 (one for each suit explained in more detail below):

  * S1 - Suit of card 1: Ordinal (1-4) representing: Hearts=1, Spades=2, Diamonds=3, Clubs=4
  * C1 - Rank of card 1: Numerical (1-13) representing: Ace=1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , Jack=11, Queen=12, King=13
  * S2 - Suit of card 2: Ordinal (1-4) representing: Hearts=1, Spades=2, Diamonds=3, Clubs=4
  * C2 - Rank of card 2: Numerical (1-13) representing: Ace=1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , Jack=11, Queen=12, King=13
  * S3 - Suit of card 3: Ordinal (1-4) representing: Hearts=1, Spades=2, Diamonds=3, Clubs=4
  * C3 - Rank of card 3: Numerical (1-13) representing: Ace=1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , Jack=11, Queen=12, King=13
  * S4 - Suit of card 4: Ordinal (1-4) representing: Hearts=1, Spades=2, Diamonds=3, Clubs=4
  * C4 - Rank of card 4: Numerical (1-13) representing: Ace=1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , Jack=11, Queen=12, King=13
  * S5 - Suit of card 5: Ordinal (1-4) representing: Hearts=1, Spades=2, Diamonds=3, Clubs=4
  * C5 - Rank of card 5: Numerical (1-13) representing: Ace=1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , Jack=11, Queen=12, King=13

CLASS Poker Hand: Ordinal (0-9)
  * 0 - Nothing in hand; not a recognized poker hand
  * 1 - One pair; one pair of equal ranks within five cards
  * 2 - Two pairs; two pairs of equal ranks within five cards
  * 3 - Three of a kind; three equal ranks within five cards
  * 4 - Straight; five cards, sequentially ranked with no gaps
  * 5 - Flush; five cards with the same suit
  * 6 - Full house; pair + different rank three of a kind
  * 7 - Four of a kind; four equal ranks within five cards
  * 8 - Straight flush; straight + flush
  * 9 - Royal flush; {Ace, King, Queen, Jack, Ten} + flush

## LOADING DATA
The training and testing data do not need the last column which represents the poker hand. We drop that column and construct trainX and testX. The trainY and testY represents the output which consists only of the last column(poker hand). The output vector is then cconverted to one hot vector since its a multiclass classification. 

## CREATING MODEL
We created a simple neural network model with Keras. A neural network is inspired by a biological neural network but here the connections between neurons are modeled by weights. Neural networks are organized into layers of nodes. An individual node might be connected to several nodes in the layer beneath it, from which it receives data, and several nodes in the layer above it, to which it sends data. First we created a sequential model which means that the output of each layer is added as input to the next layer. Adding layers are like stacking lego blocks one by one. We have used dense layer which represents a fully connected layer. We need to mention input dimension in the input layer, number of neurons in hidden layer, output dimension in the output layer. The neural netowork consists of 10 neurons in the input layer, 15 neurons in 1st and 2nd hidden layer and 10 neurons in the output layer, one for each poker hand. The activation function we used here is relu (rectified linear unit). The output layer has different activation function called softmax since this is a multiclass classification. 

## COMPILING MODEL
Model is compiled using compile() method of keras. We have to specifiy the loss function, optimizer and metrics to be used. We use Categorical_crossentropy loss since we have multiple classes (10 poker hand classes). The optimizer that we have used is adam. Optimizers are methods used to change the attributes of neural network such as weights and learning rate in order to reduce the loss. Adam is an adaptive learning rate method. Finally, we have used accuracy as metric to judge the performance of our model. model.summary() method shows the summary of the whole model including the shape of all layers.

## FITTING MODEL
model.fit() method is used to train the model with data input and label output. We can further mention number of epochs, bacth_size() as arguments to the fit method. Epochs means how many iterations we want to train the model. If the dataset is big, we can divide it by batches where each batch size equal to batch_size and train the model batch by batch. If Shuffle is argument is set to True, if shuffles the dataset before creating each batch. Verbose represents if want to see the status of each epoch or not. If verbose is et to 0, nothing will be shown, if verbose=1 or 2, each epoch's loss and accuracy will be shown. Model performance depends on diffferent parameters and hypermaters.

## EVALUATING MODEL
We can get testing or training accuracy by model.evaluate() method. This method takes data input and output as arguments. If we use any batch_size, we have to mention that here too. The evaluate method returns loss and accruacy of the model. Loss specifies how poorly or well a model behaves after each iteration of optimization. An accuracy metric measures the model's performance. Here, we see that our model achieves about 58.7% accuracy. The loss is about 90%

## TUNING MODEL
Model can be tuned in multiple ways. Here we, will show how different parameters affect a neural model's performance. We will tune our model by changing the following parameters:

 1. Number of hidden layers and neurons
 2. Batch size
 3. Number of epochs
 4. Optimizer

### Changing No. of Hidden Layers:
A new model is created with two more hidden layers. This time we will use more neurons in the hidden layers. We call this model as model2 which is sequential as before, input layer has 10 neurons, output layers has 10 neurons for each poker hand class, each hdiden layer has 50 neurons. The activaton functions are same as our first model. The batch size, optimizer and epochs are same. The model's accuracy change drastically. Our base model with one hidden layer consisting of 15 neurons has about 57% accuracy where the second model with two hidden layers each consisting of 50 neurons has about 74% accuracy.

### Changing Batch size:
Depending on the model and dataset, different parameters can have diffrent impact. Now we will see how batch size affects model's performance. Remember that our base model (model 1) has accuracy 57% with batch size 100. Here we use same hidden layers and parameters as base model except batch size which is 50 now. So, we divide our dataset into smaller batches than previous one. 

### Changing Epochs:
We create our 4th model where number of epochs is 500 and all the other parameters are same as our first model. The model's accuracy improve to 75% from 57%.

### Changing Optimizer:
A different optimizer: SGD( Stochastic Gradient Descent) optimizer is used in the 5th model which. Gradient descent method can update each parameter of a model, observe how a change would affect the objective function, choose a direction that would lower the error rate, and continue iterating until the objective function converges to the minimum. SGD is a variant of gradient descent. SGD computes on a small batch of data instead on cosidering the whole dataset. The model's accuracy improve to 75% from 57%.
