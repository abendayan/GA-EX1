from genome_handler import GenomeHandler
from ga import GA
import cPickle
import gzip

nn_param_choices = {
    'nb_neurons': [128, 64],
    'nb_layers': 2,
    'activation': ['tanh', 'sigmoid', 'ReLU']
}

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
genome_handler_lo = GenomeHandler(nn_param_choices)

num_generations = 3000
population_size = 100

ga = GA(genome_handler_lo)
ga.run(train_set, valid_set, test_set, num_generations, population_size)
