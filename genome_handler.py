import numpy as np
import random
from nn import NN as NN
import pickle
from math import sqrt

class GenomeHandler:
    def __init__(self, nn_param_choices):
        self.nn_param_choices = nn_param_choices
        random.seed(0)
        np.random.seed(0)

    def mutate(self, model, num_mutations):
        num_mutations = random.choice(range(num_mutations+1))
        # for j in range(num_mutations):
        for i in range(model.nb_layers):
            W, b = model.params[i]
            # if b_or_w:
            # x = np.random.randint(W.shape[0])
            # y = np.random.randint(W.shape[1])
            # w = W[x][y]
            # W[x][y] = random.uniform(w-0.10*w, w+0.10*w)
            # model.params[i][0] = W
            for column in range(1, model.params[i][0].shape[0]+1):
                if random.random() < 0.05:
                    model.params[i][0][column-1:column] += np.random.randn(1, W.shape[1]) / (np.sqrt(W.shape[0]))
                # model.params[i][0] += 0.005*np.random.normal(0, eps, size = W.shape)
                # divide = max(model.params[i][0].max(), abs(model.params[i][0].min()))
                # model.params[i][0] /= divide
                # model.params[i][0] * eps
            # else:
            # x = np.random.randint(b.shape[0])
            # B = b[x]
            # b[x] = random.uniform(b-0.10*b, b+0.10*b)
            # model.params[i][1] = b
            for j in range(b.shape[0]):
                if random.random() < 0.05:
                    model.params[i][1][j] += np.random.randn()/(np.sqrt(b.shape[0]))
                # divide = max(model.params[i][1].max(), abs(model.params[i][1].min()))
                # model.params[i][1] /= divide
                # model.params[i][1] * eps1
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
