#!/usr/bin/bash
import lasagne
import numpy as np
import theano
import theano.tensor as T
import os.path
import gym

class PongLearnerAgent:

    def __init__(self, x, n_visible, n_hidden):
        self.l_input = lasagne.layers.InputLayer((None, n_visible), x)
        self.l_dense1 = lasagne.layers.DenseLayer(self.l_input, n_hidden, nonlinearity = lasagne.nonlinearities.rectify)
        self.l_dense2 = lasagne.layers.DenseLayer(self.l_dense1, 2, nonlinearity = lasagne.nonlinearities.softmax)

    def get_output(self):
        return lasagne.layers.get_output(self.l_dense2)

    def get_params(self):
        return lasagne.layers.get_all_params(self.l_dense2, trainable = True)

    def save_model(self, name):
        np.save(name, lasagne.layers.get_all_param_values(self.l_dense2))

    def load_model(self, name):
        lasagne.layers.set_all_param_values(self.l_dense2, np.load(name))

    def preprocess_image(self, I):
        I = I[35:195]
        I = I[::2,::2,0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    def discount_rewards(self, r, gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0:
                running_add = 0
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r




x = T.matrix("x")
y = T.matrix("y")
discounted_rewards = T.vector("discounted_rewards")
batch_size = 10
n_visible = 80 * 80
n_hidden = 200
learning_rate = 1e-4
rho = 0.99
discount_factor = 0.99
save_freq = 1000

agent = PongLearnerAgent(x, n_visible, n_hidden)
out = agent.get_output()
params = agent.get_params()
loss = T.mean(lasagne.objectives.categorical_crossentropy(out, y))
updates = lasagne.updates.rmsprop(loss, params, learning_rate = learning_rate, rho = rho)
train_fn = theano.function([x, y], loss, updates = updates, allow_input_downcast = True)
predict = theano.function([x], out, allow_input_downcast = True)

if os.path.isfile("ponglearneragent.npy"):
    agent.load_model("ponglearneragent.npy")

env = gym.make("Pong-v0")
observation = env.reset()
episode = 0
prev_x = None
hist_xs = list()
hist_labels = list()
hist_rewards = list()
tmp_xs = list()
tmp_labels = list()
discounted_tmp_rewards = list()
reward_sum = 0
running_reward = None

while True:
    curr_x = agent.preprocess_image(observation)
    diff_x = curr_x - prev_x if prev_x is not None else np.zeros((n_visible))
    prev_x = curr_x
    prob = predict([diff_x])
    action = np.random.choice([2, 3], p = prob[0])
    label = [1, 0] if action == 2 else [0, 1]
    observation, reward, done, _ = env.step(action)
    hist_xs.append(diff_x)
    hist_labels.append(label)
    hist_rewards.append(reward)
    reward_sum += reward
    if done:
        episode += 1
        tmp_xs.append(hist_xs)
        tmp_labels.append(hist_labels)
        hist_rewards = agent.discount_rewards(np.asarray(hist_rewards), discount_factor)
        hist_rewards -= np.mean(hist_rewards)
        hist_rewards /= np.std(hist_rewards)
        discounted_tmp_rewards.append(hist_rewards.tolist())
        hist_xs = []
        hist_labels = []
        hist_rewards = []
        if episode % batch_size == 0:
            print "Appying the learning step..."
            for i in range(batch_size):
                ep_xs = np.stack(tmp_xs[i], 0)
                for j in range(len(discounted_tmp_rewards[i])):
                    tmp_labels[i][j] = [x * discounted_tmp_rewards[i][j] for x in tmp_labels[i][j]]
                ep_labels = np.stack(tmp_labels[i], 0)
                train_fn(ep_xs, ep_labels)
            tmp_xs = []
            tmp_labels = []
            discounted_tmp_rewards = []
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
#        print "Resetting environment..."
        print "Episode %d total reward was %f, running mean is %f" % (episode, reward_sum, running_reward)
        if episode % save_freq == 0:
            print "Saving the trained model to a file..."
            agent.save_model("ponglearneragent")
        reward_sum = 0
        observation = env.reset()
        prev_x = None
#    if reward != 0:
#        print "Episode %d: game finished with reward %f" % (episode, reward)
