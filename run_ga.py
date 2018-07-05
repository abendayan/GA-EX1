from genome_handler import GenomeHandler
from ga import GA
import cPickle
import gzip
import argparse
import os
from mnist import MNIST
import numpy as np
from loader import load_data
from nn import NN, load_model

nn_param_choices = {
    'nb_neurons': [200, 200],
    'nb_layers': 2,
    'activation': ['tanh', 'sigmoid', 'ReLU']
}

parser = argparse.ArgumentParser(description='Genetical Algorithm')

parser.add_argument('-model', default='None')
parser.add_argument('-images', default='t10k-images-idx3-ubyte')
parser.add_argument('-labels', default='t10k-labels-idx1-ubyte')
result = parser.parse_args()

train_image, train_labels, valid_image, valid_labels, test_image, test_labels = load_data(result.images, result.labels)

genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 20000
population_size = 100

use_model = not result.model == 'None'
if use_model:
    ga_model = load_model(result.model)
else:
    ga = GA(genome_handler_lo)
    ga_model = ga.run((train_image, train_labels), (valid_image, valid_labels), (test_image, test_labels), num_generations, population_size)


accu = ga_model.validate_batch(test_image, test_labels, 64)[0]
print "Test accuracy is {0}".format(accu)
labels_test = ga_model.predict(test_image)[0]
write_file = open("test_ga.pred", "w")
to_write = ""
for label in labels_test:
    to_write += str(label) + "\n"
write_file.write(to_write)
write_file.close
