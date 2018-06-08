import numpy as np
import random
from nn import NN as NN
import pickle
from math import sqrt

class GenomeHandler:
    def __init__(self, nn_param_choices):
        self.nn_param_choices = nn_param_choices
        random.seed(0)

    def mutate(self, model, num_mutations):
        num_mutations = random.choice(range(num_mutations+1))
        for j in range(num_mutations):
            for i in range(model.nb_layers):
                W, b = model.params[i]
                eps = sqrt(6.0/(W.shape[0] + W.shape[1]))
                eps1 = sqrt(6.0/(b.shape[0]))
                model.params[i][0] -= 0.005*np.random.uniform(low=-eps, high=eps, size=W.shape)
                # import pdb; pdb.set_trace()
                # divide = max(model.params[i][0].max(), abs(model.params[i][0].min()))
                # model.params[i][0] /= divide
                # model.params[i][0] = W + 0.001*np.random.uniform(low=-eps, high=eps, size=W.shape)
                model.params[i][1] -= 0.005*np.random.uniform(low=-eps, high=eps, size=b.shape)
            # Mutate one of the params.
            # if mutation == 'nb_neurons':
            #     index = random.choice(range(len(network['nb_neurons'])))
            #     network['nb_neurons'][index] = random.choice(self.nn_param_choices['nb_neurons'])
            # elif mutation == 'nb_layers':
            #     network[mutation] = random.choice(self.nn_param_choices[mutation])
            #     if network['nb_layers'] > len(network['nb_neurons']):
            #         for _ in range(network['nb_layers'] - len(network['nb_neurons'])):
            #             network['nb_neurons'].append(random.choice(self.nn_param_choices['nb_neurons']))
            #     elif network['nb_layers'] < len(network['nb_neurons']):
            #         network['nb_neurons'] = network['nb_neurons'][:-network['nb_layers']]
            # else:
            #     network[mutation] = random.choice(self.nn_param_choices[mutation])
        return model

    def decode(self, genome):
        dim = [784]
        for neurons in genome['nb_neurons']:
            dim.append(neurons)
        dim.append(10)
        model = NN(genome['activation'], dim)
        return model

    # def genome_representation(self):
    #     encoding = []
    #     for i in range(self.convolution_layers):
    #         for key in self.convolutional_layer_shape:
    #             encoding.append("Conv" + str(i) + " " + key)
    #     for i in range(self.dense_layers):
    #         for key in self.dense_layer_shape:
    #             encoding.append("Dense" + str(i) + " " + key)
    #     encoding.append("Optimizer")
    #     return encoding

    def generate(self):
        network = {}
        for key in self.nn_param_choices:
            network[key] = self.nn_param_choices[key]
        network['nb_neurons'] = []
        for i in range(network['nb_layers']):
            network['nb_neurons'].append(self.nn_param_choices['nb_neurons'][i])
        return self.decode(network)

    # metric = accuracy or loss
    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def load_model(self, models_file):
        return pickle.load(open(models_file))
