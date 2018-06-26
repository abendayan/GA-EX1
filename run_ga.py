from genome_handler import GenomeHandler
from ga import GA
import cPickle
import gzip
import argparse
import os
from mnist import MNIST
import numpy as np
from loader import load_data

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

train_image, train_labels, valid_image, valid_labels, test_image, test_labels = load_data(result.images, result.labels)

genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 7000
population_size = 100

ga = GA(genome_handler_lo)
ga.run((train_image, train_labels), (valid_image, valid_labels), (test_image, test_labels), num_generations, population_size)
