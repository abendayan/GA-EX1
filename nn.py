# Adele Bendayan 336141056
import sys
import cPickle, gzip
import time
import pdb
from random import shuffle
from math import log, sqrt

EPOCHS = 100
BATCH_SIZE = 50
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
        x[x == 0] = 0.01
    else:
        x = np.maximum(x, 0.01*x)
    return x

def tanh(x, derivate = False):
    if derivate:
        x = 1 - np.power(x, 2)
    else:
        x = np.tanh(x)
    return x

class NN:
    def __init__(self, learning_rate, activation_function, input_dim, hidden_dim, output_dim):
        self.activation_function = activation_function
        np.random.seed(0)
        self.lr = learning_rate
        eps = sqrt(6.0/(input_dim + hidden_dim))
        self.W1 = (np.random.rand(input_dim, hidden_dim) - .5) * .1
        # np.random.uniform(low=-eps, high=eps, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        eps = sqrt(6.0/(output_dim + hidden_dim))
        self.W2 = np.random.uniform(low=-eps, high=eps, size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    def predict(self, x):
        probs = self.forward(x)
        labels = np.argmax(probs, axis=1)
        return labels

    def validate(self, x, y):
        labels = self.predict(x)
        good = 0.0
        bad = 0.0
        for (label, true_label) in zip(labels, y):
            if label == int(true_label):
                good += 1
            else:
                bad += 1
        return good/(good+bad)

    def forward(self, x):
        z1 = x.dot(self.W1) + self.b1
        a1 = self.activation_function(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.array([softmax(z) for z in z2])
        return a2

    def first_layer(self, x):
        z1 = x.dot(self.W1) + self.b1
        a1 = self.activation_function(z1)
        return a1

    def forward_backward(self, x, Y):
        y_hot = np.array([one_hot_vector(int(y), CLASSES) for y in Y])
        a1 = self.first_layer(x)
        a2 = self.forward(x)

        gb2 = a2 - y_hot
        gW2 = a1.T.dot(gb2)
        gb1 = gb2.dot(self.W2.T) * self.activation_function(a1, True)
        gW1 = x.T.dot(gb1)
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1.sum(axis=0)
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2.sum(axis=0)
        loss = 0.0
        for i in range(len(x)):
            loss -= log(a2[i][int(Y[i])])
        return loss

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

train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")
randomize = np.arange(len(train_x))
np.random.shuffle(randomize)
train_x = train_x[randomize]
train_y = train_y[randomize]
train_x_train = train_x[0:len(train_x)*80/100]
train_y_train = train_y[0:len(train_y)*80/100]
train_x_valid = train_x[len(train_x)*80/100+1:-1]
train_y_valid = train_y[len(train_y)*80/100+1:-1]

print "Finish getting the data {0}".format(passed_time(start_time))
lr = 0.01
multiNN = NN(lr, sigmoid, len(train_x_train[0]), 200, CLASSES)
size_training = len(train_x_train)
train_x_train = normalize(train_x_train)
train_x_valid = normalize(train_x_valid)
test_x = normalize(test_x)
print "Start training!"
for epoch in range(EPOCHS):
    randomize = np.arange(size_training)
    np.random.shuffle(randomize)
    X_train = train_x_train[randomize]
    Y_train = train_y_train[randomize]
    loss = 0.0
    for i in range(0, len(train_x_train), BATCH_SIZE):
        loss += multiNN.forward_backward(X_train[i:i+BATCH_SIZE], Y_train[i:i+BATCH_SIZE])
    print "Loss for epoch {0} is {1}".format(epoch, loss/size_training)
    # if epoch % 10 == 0:
    accu = multiNN.validate(train_x_valid, train_y_valid)
    print "Accuracy after epoch {0} is {1}".format(epoch, accu)
    print "Done in {0}".format(passed_time(start_time))
    if accu > 0.9:
        break
# print "Accuracy after training is {0}".format(multiNN.validate(train_x_valid, train_y_valid))
# print "Done learning in {0}".format(passed_time(start_time))
labels_test = multiNN.predict(test_x)
write_file = open("test.pred", "w")
to_write = ""
for label in labels_test:
    to_write += str(label) + "\n"
write_file.write(to_write)
write_file.close
