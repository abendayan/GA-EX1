from __future__ import print_function

from genome_handler import GenomeHandler
from ga import GA
import cPickle, gzip

nn_param_choices = {
    'nb_neurons': [64, 128, 256, 512, 768, 1024],
    'nb_layers': [1, 2, 3, 4],
    'activation': ['ReLu', 'tanh', 'sigmoid'],
    'batch': [1, 4, 8, 12]
}

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 10
population_size = 10
num_epochs = 1

ga = GA(genome_handler_lo, 'genomes.csv')
model = ga.run(train_set, valid_set, test_set, num_generations, population_size, num_epochs)
model.summary()
