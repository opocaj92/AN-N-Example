import lasagne
import theano.tensor as T
import theano
import numpy as np
import os

class ChaRNN:
    def __init__(self, num_chars, num_hidden_units, x, seq_length, grad_clip):
        self.l_input = lasagne.layers.InputLayer((None, seq_length, num_chars), x)
        self.l_rec1 = lasagne.layers.LSTMLayer(self.l_input, num_hidden_units, nonlinearity = lasagne.nonlinearities.tanh, grad_clipping = grad_clip)
        self.l_rec2 = lasagne.layers.LSTMLayer(self.l_rec1, num_hidden_units, nonlinearity = lasagne.nonlinearities.tanh, grad_clipping = grad_clip)
        self.l_out = lasagne.layers.DenseLayer(self.l_rec2, num_chars, nonlinearity = lasagne.nonlinearities.softmax)

    def get_output(self):
        return lasagne.layers.get_output(self.l_out)

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable = True)

    def save_model(self, name):
        np.save(name, lasagne.layers.get_all_param_values(self.l_out))

    def load_model(self, name):
        lasagne.layers.set_all_param_values(self.l_out, np.load(name))

def one_hot(c, n):
    ohv = np.zeros(n)
    ohv[c] = 1
    return np.asarray(ohv)

hidden_units = 512
learning_rate = 1e-2
seq_length = 20
batch_size = 128
grad_clip = 100
num_epochs = 50
print_freq = 1000
sample_size = 200
generation_phrase = "The quick brown fox jumps"

x = T.itensor3("x")
y = T.imatrix("y")

data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'Data has %d characters, %d unique' % (data_size, vocab_size)
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

net = ChaRNN(vocab_size, hidden_units, x, seq_length, grad_clip)
print "Network created..."
out = net.get_output()
params = net.get_params()
loss = T.mean(lasagne.objectives.categorical_crossentropy(out, y))
updates = lasagne.updates.adagrad(loss, params, learning_rate = learning_rate)
print "Updates created..."
train_fn = theano.function([x, y], loss, updates = updates, allow_input_downcast = True)
predict = theano.function([x], out, allow_input_downcast = True)
print "Train functions created..."

if os.path.isfile("charnn.npy"):
    net.load_model("charnn.npy")

p, avg_loss = 0, 0
print "Start training..."
for n in range(num_epochs * data_size / batch_size):
    if p + batch_size + seq_length >= len(data):
        print "Start again..."
        p = 0
    inputs = np.array([[one_hot(char_to_ix[ch], vocab_size) for ch in data[p + b:p + b + seq_length]] for b in range(batch_size)])
    targets = np.array([one_hot(char_to_ix[ch], vocab_size) for ch in data[p + seq_length:p + batch_size + seq_length]])
    avg_loss += train_fn(inputs, targets)
    p += batch_size
    if n % print_freq == 0:
        print 'Iteration %d -> average loss: %f' % (n, avg_loss / print_freq)
        avg_loss = 0
        assert len(generation_phrase) >= seq_length
        txt = list(generation_phrase)[:seq_length]
        char = [one_hot(char_to_ix[ch], vocab_size) for ch in txt]
        txt = ""
        for i in range(sample_size):
            prob = predict([char])
            l = np.random.choice(np.arange(vocab_size), p = prob[0])
            char = np.append(char[1:], [one_hot(l, vocab_size)], axis = 0)
            txt += ix_to_char[l]
        print '----\n %s \n----' % (txt)
        net.save_model("charnn.npy")
net.save_model("charnn.npy")
print "End of training!"
