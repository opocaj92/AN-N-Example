import lasagne
from mnist import MNIST
import numpy as np
import theano
import theano.tensor as T
import time
import os.path

x = T.tensor4("x")
y = T.ivector("y")
epochs = 10
batch_size = 200

l_input = lasagne.layers.InputLayer((None, 1, 28, 28), x)
l_conv1 = lasagne.layers.Conv2DLayer(l_input, 30, (5, 5), nonlinearity = lasagne.nonlinearities.rectify)
l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, (2, 2))
l_conv2 = lasagne.layers.Conv2DLayer(l_pool1, 15, (3, 3), nonlinearity = lasagne.nonlinearities.rectify)
l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, (2, 2))
l_drop = lasagne.layers.DropoutLayer(l_pool2, 0.2)
l_dense1 = lasagne.layers.DenseLayer(l_drop, 128, nonlinearity = lasagne.nonlinearities.rectify)
l_dense2 = lasagne.layers.DenseLayer(l_dense1, 50, nonlinearity = lasagne.nonlinearities.rectify)
l_dense3 = lasagne.layers.DenseLayer(l_dense2, 10, nonlinearity = lasagne.nonlinearities.softmax)

out = lasagne.layers.get_output(l_dense3)
params = lasagne.layers.get_all_params(l_dense3, trainable = True)
loss = T.mean(lasagne.objectives.categorical_crossentropy(out, y))
updates = lasagne.updates.adam(loss, params, learning_rate = 0.01)
train_fn = theano.function([x, y], loss, updates = updates, allow_input_downcast = True)

predict = lasagne.layers.get_output(l_dense3, deterministic = True)
accuracy = T.mean(T.eq(T.argmax(predict, axis = 1), y), dtype = theano.config.floatX)
val_fn = theano.function([x, y], accuracy, allow_input_downcast = True)

mndata = MNIST("./mnist_dataset")
training, train_labels = mndata.load_training()
training = np.asarray(training).reshape(len(training), 1, 28, 28) / np.float(255)
train_labels = np.asarray(train_labels)
test, test_labels = mndata.load_testing()
test = np.asarray(test).reshape(len(test), 1, 28, 28) / np.float(255)
test_labels = np.asarray(test_labels)

if os.path.isfile("myconvnet.npy"):
    print "Loading the model from file..."
    lasagne.layers.set_all_param_values(l_dense3, np.load("myconvnet.npy"))
else:
    print "Starting training..."
    for i in range(epochs):
        err = 0
        n_batches = 0
        start_time = time.time()
        for j in range(0, len(training), batch_size):
            err += train_fn(training[j:j + batch_size], train_labels[j:j + batch_size])
            n_batches += 1
        print "Epoch {}/{} took {:.3f}s to complete with an average loss of {:.6f}".format(i + 1, epochs, time.time() - start_time, err / n_batches)
    print "Training completed..."
    print "Saving the trained model to a file..."
    np.save("myconvnet", lasagne.layers.get_all_param_values(l_dense3))

score = 0
n_batches = 0
for j in range(0, len(test), batch_size):
    score += val_fn(test[j:j + batch_size], test_labels[j:j + batch_size])
    n_batches += 1
print
print "CNN accuracy over the test set is {:.2f}%".format((float(score) / n_batches) * 100)

