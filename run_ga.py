from genome_handler import GenomeHandler
from ga import GA
import cPickle
import gzip
import argparse
import os
from mnist import MNIST
import numpy as np

def normalize(x):
    return x.astype(np.float)/255.0

nn_param_choices = {
    'nb_neurons': [200, 100],
    'nb_layers': 2,
    'activation': ['tanh', 'sigmoid', 'ReLU']
}

parser = argparse.ArgumentParser(description='Genetical Algorithm')

# parser.add_argument('-model', default='None')
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
test_labels = np.array(test_labels)
test_image = normalize(np.array(test_image))

genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 7000
population_size = 100

ga = GA(genome_handler_lo)
ga.run((train_image, train_labels), (valid_image, valid_labels), (test_image, test_labels), num_generations, population_size)
