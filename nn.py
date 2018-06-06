# Adele Bendayan 336141056
import sys
import cPickle, gzip
import numpy as np
import time
import pdb
from random import shuffle
from math import log, sqrt

EPOCHS = 100
BATCH_SIZE = 12
CLASSES = 10

start_time = time.time()

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def softmax(x):
    m = np.max(x)
    e = np.exp(x - m)  # prevent overflow
    return e / np.sum(e, axis=0)

def sigmoid(x, derivate = False):
    if derivate:
        x = x * (1.0 - x)
    else:
        x = 1 / (1 + np.exp(-x))
    return x

def one_hot_vector(i, size):
    vector = np.zeros(size)
    vector[i] = 1
    return vector

def ReLU(x, derivate = False):
    if derivate:
        x = 1. * (x > 0)
        x[x == 0] = 0
    else:
        x = np.maximum(x, 0)
    return x

def tanh(x, derivate = False):
    if derivate:
        x = 1 - np.power(x, 2)
    else:
        x = np.tanh(x)
    return x

class NN:
    def __init__(self, learning_rate, activation_function, dim):
        self.activation_function = activation_function
        np.random.seed(0)
        self.lr = learning_rate
        self.create_classifier(dim)
        # eps = sqrt(6.0/(input_dim + hidden_dim))
        # self.W1 = (np.random.rand(input_dim, hidden_dim) - .5) * .1
        # self.b1 = np.zeros((1, hidden_dim))
        # eps = sqrt(6.0/(output_dim + hidden_dim))
        # self.W2 = np.random.uniform(low=-eps, high=eps, size=(hidden_dim, output_dim))
        # self.b2 = np.zeros((1, output_dim))

    def create_classifier(self, dims):
        print dims
        self.params = []
        for i in range(len(dims) - 1):
            W = (np.random.rand(dims[i], dims[i + 1]) - .5) * .1
            b = np.zeros(dims[i + 1])
            self.params.append([W, b])

    def classifier_output(self, x, predict = False):
        for i in range(len(self.params) - 1):
            W, b = self.params[i]
            x = self.activation_function(np.dot(x, W) + b)
        W, b = self.params[-1]
        z2 = np.dot(x, W) + b
        if predict:
            probs = softmax(z2)
        else:
            probs = np.array([softmax(z) for z in z2])
        return probs

    def predict(self, x):
        return np.argmax(self.classifier_output(x, True))

    def validate(self, x, y):
        good = bad = 0.0
        for (data, true_label) in zip(x, y):
            if self.predict(data) == true_label:
                good += 1
            else:
                bad += 1
        return good / (good + bad)


    def loss_and_gradients(self, x, Y):
        U, b_tag = self.params[-1]

        y_hot = np.array([one_hot_vector(int(y), U.shape[1]) for y in Y])
        loss = 0.0
        y_tag = self.classifier_output(x)
        for i in range(len(x)):
            loss -= log(y_tag[i][int(Y[i])])

        gradients = []

        list_z = [x]
        list_a = [x]

        for W, b in self.params:
            list_z.append(list_z[-1].dot(W) + b)
            list_a.append(self.activation_function(list_z[-1]))

        diff = y_tag - y_hot
        gradients.insert(0, [np.array(list_a[-2]).transpose().dot(diff), diff])

        for i in (range(len(self.params) - 1))[::-1]:
            gb = (self.params[i + 1][0].dot(gradients[0][1].T)).T * self.activation_function(list_a[i+1], True)
            gW = np.array(list_z[i]).transpose().dot(np.array(gb))
            gradients.insert(0, [gW, gb])

        return loss, gradients

    def forward_backward(self, x, Y):
        loss, grads = self.loss_and_gradients(x, Y)
        for i in range(len(self.params)):
            self.params[i][0] -= grads[i][0] * self.lr
            self.params[i][1] -= grads[i][1].sum(axis=0) * self.lr
        train_loss = loss/len(x)
        return train_loss

def to_batch(X, Y):
    batched_X = []
    batched_Y = []
    for i in range(int(np.ceil(len(X)/BATCH_SIZE))):
        batched_X.append(X[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        batched_Y.append(Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    return batched_X, batched_Y

def normalize(x):
    max_x = 255
    middle_x = 125
    x -= middle_x
    x /= (max_x - middle_x)
    return x

print "Start getting the data {0}".format(passed_time(start_time))
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
# import pdb; pdb.set_trace()
# np.random.shuffle(train_set)
# np.random.shuffle(valid_set)

print "Finish getting the data {0}".format(passed_time(start_time))
lr = 0.01
multiNN = NN(lr, ReLU, [len(train_set[0][0]), 200, 200, CLASSES])
size_training = len(train_set[0])
# train_x_train = normalize(train_set[0])
# train_x_valid = normalize(train_x_valid)
# test_x = normalize(test_x)
print "Start training!"
for epoch in range(EPOCHS):
    # randomize = np.arange(size_training)
    # np.random.shuffle(train_set)
    # X_train = train_x_train[randomize]
    # Y_train = train_y_train[randomize]
    loss = 0.0
    for i in range(0, len(train_set[1]), BATCH_SIZE):
        loss += multiNN.forward_backward(train_set[0][i:i+BATCH_SIZE], train_set[1][i:i+BATCH_SIZE])
    print "Loss for epoch {0} is {1}".format(epoch, loss/size_training)
    # if epoch % 10 == 0:
    accu = multiNN.validate(valid_set[0], valid_set[1])
    print "Accuracy after epoch {0} is {1}".format(epoch, accu)
    print "Done in {0}".format(passed_time(start_time))
# print "Accuracy after training is {0}".format(multiNN.validate(train_x_valid, train_y_valid))
# print "Done learning in {0}".format(passed_time(start_time))
# labels_test = multiNN.predict(test_x)
# write_file = open("test.pred", "w")
# to_write = ""
# for label in labels_test:
#     to_write += str(label) + "\n"
# write_file.write(to_write)
# write_file.close
