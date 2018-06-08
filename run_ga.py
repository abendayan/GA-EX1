# from __future__ import print_function

from genome_handler import GenomeHandler
from ga import GA
import cPickle, gzip
import numpy as np

nn_param_choices = {
    'nb_neurons': [256, 128],
    'nb_layers': 2,
    'activation': 'tanh'
}

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
print(np.all(train_set ==0))
genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 1000
population_size = 100
num_epochs = 1

ga = GA(genome_handler_lo, 'genomes.csv')
layers, neurons, activation = ga.run(train_set, valid_set, test_set, num_generations, population_size, num_epochs)
print "Number layers: " + str(layers)
print "Number neurons: "
i = 1
for neuron in neurons:
    print "Layer number ", i, " neurons: ", neuron
    i += 1
print "Activation function: " + str(activation)
