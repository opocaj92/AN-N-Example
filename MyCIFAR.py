import cPickle
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import os.path

def unpickle(filename):
    with open(filename, 'rb') as fo:
        d = cPickle.load(fo)
    return d

d = [unpickle("cifar-10-batches-py/data_batch_" + str(i)) for i in range(1, 6)]
training = reduce((lambda x, y: np.concatenate((x, y), axis = 0)), map((lambda x: x['data']), d))
training = training.reshape(-1, 3, 32, 32) / np.float(255)
train_labels = reduce((lambda x, y: np.concatenate((x, y), axis = 0)), map((lambda x: x['labels']), d))

d = unpickle("cifar-10-batches-py/test_batch")
test = d['data'].reshape(-1, 3, 32, 32) / np.float(255)
test_labels = np.array(d['labels'], dtype = np.uint8)

x = T.tensor4("x")
y = T.ivector("y")
epochs = 10000
batch_size = 32

l_input = lasagne.layers.InputLayer((None, 3, 32, 32), x)
l_conv1 = lasagne.layers.Conv2DLayer(l_input, 32, (3, 3), pad = "same", nonlinearity = lasagne.nonlinearities.rectify)
l_conv2 = lasagne.layers.Conv2DLayer(l_conv1, 32, (3, 3), pad = "same", nonlinearity = lasagne.nonlinearities.rectify)
l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv2, (2, 2))
l_drop1 = lasagne.layers.DropoutLayer(l_pool1, 0.25)
l_conv3 = lasagne.layers.Conv2DLayer(l_drop1, 64, (3, 3), pad = "same", nonlinearity = lasagne.nonlinearities.rectify)
l_conv4 = lasagne.layers.Conv2DLayer(l_conv3, 64, (3, 3), pad = "same", nonlinearity = lasagne.nonlinearities.rectify)
l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv4, (2, 2))
l_drop2 = lasagne.layers.DropoutLayer(l_pool2, 0.25)
l_flat = lasagne.layers.FlattenLayer(l_drop2)
l_dense1 = lasagne.layers.DenseLayer(l_conv3, 512, nonlinearity = lasagne.nonlinearities.rectify)
l_drop3 = lasagne.layers.DropoutLayer(l_dense1, 0.5)
l_dense2 = lasagne.layers.DenseLayer(l_drop3, 10, nonlinearity = lasagne.nonlinearities.softmax)

out = lasagne.layers.get_output(l_dense2)
params = lasagne.layers.get_all_params(l_dense2, trainable = True)
loss = T.mean(lasagne.objectives.categorical_crossentropy(out, y))
updates = lasagne.updates.adam(loss, params, learning_rate = 1e-4)
train_fn = theano.function([x, y], loss, updates = updates, allow_input_downcast = True)

predict = lasagne.layers.get_output(l_dense2, deterministic = True)
accuracy = T.mean(T.eq(T.argmax(predict, axis = 1), y), dtype = theano.config.floatX)
val_fn = theano.function([x, y], accuracy, allow_input_downcast = True)

if os.path.isfile("mycifar.npy"):
    print "Loading the model from file..."
    lasagne.layers.set_all_param_values(l_dense3, np.load("mycifar.npy"))
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
    np.save("mycifar", lasagne.layers.get_all_param_values(l_dense3))

score = 0
n_batches = 0
for j in range(0, len(test), batch_size):
    score += val_fn(test[j:j + batch_size], test_labels[j:j + batch_size])
    n_batches += 1
print
print "CNN accuracy over the test set is {:.2f}%".format((float(score) / n_batches) * 100)
