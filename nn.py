import cPickle
import gzip
import numpy as np
import time
from math import log, sqrt
import pickle
from mnist import MNIST
import sys
import argparse
import os

EPOCHS = 60
BATCH_SIZE = 1
CLASSES = 10

start_time = time.time()


def passed_time(previous_time):
    return round(time.time() - previous_time, 3)


def softmax(x):
    m = np.max(x)
    e = np.exp(x - m)  # prevent overflow
    return e / np.sum(e, axis=0)


def one_hot_vector(i, size):
    vector = np.zeros(size)
    vector[i] = 1
    return vector


class Activation:

    def sigmoid(self, x, derivative=False):
        if derivative:
            x = x * (1.0 - x)
        else:
            x = 1 / (1 + np.exp(-x))
        return x

    def ReLU(self, x, derivative=False):
        if derivative:
            x = 1. * (x > 0)
            x[x == 0] = 0
        else:
            x = np.maximum(x, 0)
        return x

    def tanh(self, x, derivative=False):
        if derivative:
            x = 1 - np.power(x, 2)
        else:
            x = np.tanh(x)
        return x


class NN:
    def __init__(self, activation_function, dim, learning_rate=0.001):
        self.params = None
        self.activation_function = getattr(Activation(), activation_function)
        self.lr = learning_rate
        self.create_classifier(dim)
        self.nb_layers = len(dim) - 2
        self.nb_neurons = dim[1:len(dim) - 1]
        self.activation = activation_function

    def create_classifier(self, dims):
        self.params = []
        for i in range(len(dims) - 1):
            w = self.create_w((dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.params.append([w, b])

    @staticmethod
    def create_w(dims):
        eps = sqrt(6.0 / (dims[0] + dims[1]))
        return np.random.uniform(low=-eps, high=eps, size=(dims[0], dims[1]))

    def save(self, name):
        dim = [784] + self.nb_neurons
        pickle.dump((self.params, [self.activation, dim, self.lr]), open(name, "wb"))

    def classifier_output(self, x, predict=False):
        for i in range(len(self.params) - 1):
            w, b = self.params[i]
            x = self.activation_function(np.dot(x, w) + b)
        w, b = self.params[-1]
        z2 = np.dot(x, w) + b
        if predict:
            probs = softmax(z2)
        else:
            probs = np.array([softmax(z) for z in z2])
        return probs

    def predict(self, x):
        outputs = self.classifier_output(x)
        return [np.argmax(output) for output in outputs], outputs

    def validate(self, x, y):
        good = bad = 0.0
        labels = self.predict(x)[0]
        for (label, true_label) in zip(labels, y):
            if label == true_label:
                good += 1
            else:
                bad += 1
        return good / (good + bad)

    def validate_batch(self, x, y, batch_size):
        good = bad = 0.0
        loss = 0.0
        for i in range(0, len(y), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            labels, outputs = self.predict(x_batch)
            for (output, label) in zip(outputs, y_batch):
                if output[label] > 0:
                    loss -= log(output[label])
                else:
                    loss = float("inf")
            for (label, true_label) in zip(labels, y_batch):
                if label == true_label:
                    good += 1
                else:
                    bad += 1
        return (good / (good + bad)), loss

    def loss_and_gradients(self, x, y):
        u, b_tag = self.params[-1]

        y_hot = np.array([one_hot_vector(int(Y), u.shape[1]) for Y in y])
        loss = 0.0
        y_tag = self.classifier_output(x)
        for i in range(len(x)):
            loss -= log(y_tag[i][int(y[i])])

        gradients = []

        list_z = [x]
        list_a = [x]

        for W, b in self.params:
            list_z.append(list_z[-1].dot(W) + b)
            list_a.append(self.activation_function(list_z[-1]))

        diff = y_tag - y_hot
        gradients.insert(0, [np.array(list_a[-2]).transpose().dot(diff), diff])

        for i in (range(len(self.params) - 1))[::-1]:
            gb = (self.params[i + 1][0].dot(gradients[0][1].T)).T * self.activation_function(list_a[i + 1], True)
            gW = np.array(list_z[i]).transpose().dot(np.array(gb))
            gradients.insert(0, [gW, gb])

        return loss, gradients

    def forward_backward(self, x, y):
        loss, grads = self.loss_and_gradients(x, y)
        for i in range(len(self.params)):
            self.params[i][0] -= grads[i][0] * self.lr
            self.params[i][1] -= grads[i][1].sum(axis=0) * self.lr
        train_loss = loss / len(x)
        return train_loss


def to_batch(x, y):
    batched__x = []
    batched__y = []
    for i in range(int(np.ceil(len(x) / BATCH_SIZE))):
        batched__x.append(x[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        batched__y.append(y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
    return batched__x, batched__y


def normalize(x):
    return x.astype(np.float)/255.0

def load_model(name):
    params, args = pickle.load(open(name,'r'))
    act, dim, lr = args
    model = NN(act, dim, lr)
    model.params = params
    return model

if __name__ == '__main__':
    print "Start getting the data {0}".format(passed_time(start_time))
    parser = argparse.ArgumentParser(description='Backpropagation')

    parser.add_argument('-model', default='None')
    parser.add_argument('-images', default='t10k-images-idx3-ubyte')
    parser.add_argument('-labels', default='t10k-labels-idx1-ubyte')
    result = parser.parse_args()

    mndata = MNIST('data/', return_type="numpy")
    mndata.gz = True
    train_image, train_labels = mndata.load_training()
    train_image = normalize(train_image)
    randomize = np.arange(len(train_image))
    np.random.shuffle(randomize)
    train_image = train_image[randomize]
    train_labels = train_labels[randomize]
    valid_image = train_image[:10000]
    valid_labels = train_labels[:10000]
    train_image = train_image[10000-1:-1]
    train_labels = train_labels[10000-1:-1]


    test_image, test_labels = mndata.load(os.path.join('data/', result.images), os.path.join('data/', result.labels))
    test_labels = normalize(np.array(test_labels))
    test_image = np.array(test_image)

    print "Finish getting the data {0}".format(passed_time(start_time))
    lr = 0.001
    use_model = not result.model == 'None'
    if use_model:
        multiNN = load_model(result.model)
    else:
        multiNN = NN('ReLU', [len(train_image[0]), 200, 100, CLASSES], lr)
        size_training = len(train_image)
        print "Start training!"
        for epoch in range(EPOCHS):
            randomize = np.arange(size_training)
            np.random.shuffle(randomize)
            X_train = train_image[randomize]
            Y_train = train_labels[randomize]
            loss = 0.0
            for i in range(0, len(Y_train), BATCH_SIZE):
                loss += multiNN.forward_backward(X_train[i:i + BATCH_SIZE], Y_train[i:i + BATCH_SIZE])
            print "Loss for epoch {0} is {1}".format(epoch, loss / size_training)
            accu = multiNN.validate_batch(valid_image, valid_labels, 64)[0]
            print "Accuracy after epoch {0} is {1}".format(epoch, accu)
            print "Done in {0}".format(passed_time(start_time))

    accu = multiNN.validate_batch(test_image, test_labels, 64)[0]
    print "Test accuracy is {0}".format(accu)
    labels_test = multiNN.predict(test_image)[0]
    write_file = open("test_nn.pred", "w")
    to_write = ""
    for label in labels_test:
        to_write += str(label) + "\n"
    write_file.write(to_write)
    write_file.close
    if not use_model:
        multiNN.save("model_from_nn.model")
