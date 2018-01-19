# AN(N) Example

## Some pretty easy (but still awesome) example with artificial neural networks

### What is it
This repository is just a collection of some simple examples I've developed during my studies about artificial neural networks (ANN). They are really nothing special, but could give you an interesting overview about how poverfull this class of machine learning model can be. All the examples are written in Python, using the Lasagne library (as well as Theano and Numpy at a lower level) to implement the networks. More examples will be added in future (maybe)!

### Examples 1: MNIST handwritten characters recongnition with CNNs
This simple code (*MyConvNet.py*) uses a convolutional neural network (CNN) to recognize handwritten characters in the world-famous MNIST dataset. I have used a little package named python-mnist to easily load and use the MNIST dataset. Network parameters like the number of filters in each convolutional layer or the learning rate are taken from already existing implementations. After the training process, the model is exported to a file (*myconvnet.npy*), so that it can be reloaded in the future to avoid training or can be used by another script or software. This code is written to be a clear and easy-to-understand example of how to use a CNN for image classification, as well as a basic example of ANN training with Theano and Lasagne, nothing that can be compared to the state-of-art models for MNIST recognition.

This is the output of a test run:

```
Starting training...
Epoch 1/10 took 64.460s to complete with an average loss of 0.212160
Epoch 2/10 took 64.709s to complete with an average loss of 0.079537
Epoch 3/10 took 65.185s to complete with an average loss of 0.069581
Epoch 4/10 took 65.097s to complete with an average loss of 0.057908
Epoch 5/10 took 65.638s to complete with an average loss of 0.057157
Epoch 6/10 took 67.091s to complete with an average loss of 0.051319
Epoch 7/10 took 71.583s to complete with an average loss of 0.050045
Epoch 8/10 took 70.299s to complete with an average loss of 0.054400
Epoch 9/10 took 71.259s to complete with an average loss of 0.046308
Epoch 10/10 took 67.784s to complete with an average loss of 0.043718
Training completed...
Saving the trained model to a file...

CNN accuracy over the test set is 98.51%
```

### Example 2: Atari Pong game with deep reinforcement learning
Taking inspiration from the world-famous blog post by Dr. Andrej Karpathy (read it, it is very interesting! http://karpathy.github.io/2016/05/31/rl/), I developed my own version of this deep reinforcement learning agent capable of learning the Atari Pong game (*PongLearnerAgent.py*). The OpenAI Gym framework for the Pong game environment is required. It uses a basic version of a policy gradient method (called REIFORCE) that feed the network with an estimation of the reinforcement signal based on the reward obtained while interacting with the environment at every time step. The network this way adjusts the probabilities it gives to each action (UP and DOWN in the Pong game) in each situation in order to maximize the cumulative reward it is going to obtain. Again, after the training process, the model is exported to a file (*ponglearneragent.npy*) that can be reloaded to play Pong against the OpenAI Gym provided AI (at this level of training it consistently outperform the OpenAI agent) or for further training.

### Example 3: Character-level language model with LSTMs
This example (*ChaRNN.py*) uses a two hidden layers long-short term memory (LSTM) to learn a model of the English language character by character. The input file (*input.txt*) contains the whole Shakespeare production in English, and the network, reading through it a lot of times, is able to learn how to put characters one after another (so learning word and punctuation, as well as how to concatenate them to form a phrase) to resemble Shakespeare's writing style. This time a saved model is not provided (I wasn't able to finish an entire session of training due to my bad PC, this model is not exactly lightweight to run without a GPU).

### Author
*Castellini Jacopo*
