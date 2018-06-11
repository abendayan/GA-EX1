import numpy as np
import random
from nn import NN as NN
import pickle


class GenomeHandler:
    def __init__(self, nn_param_choices):
        self.nn_param_choices = nn_param_choices
        random.seed(0)
        np.random.seed(0)

    @staticmethod
    def mutate(model):
        for i in range(model.nb_layers):
            w, b = model.params[i]
            for column in range(1, model.params[i][0].shape[0]+1):
                if random.random() < 0.05:
                    model.params[i][0][column-1:column] += np.random.randn(1, w.shape[1]) / (np.sqrt(w.shape[0]))
            for j in range(b.shape[0]):
                if random.random() < 0.05:
                    model.params[i][1][j] += np.random.randn()/(np.sqrt(b.shape[0]))
        return model

    @staticmethod
    def decode(genome):
        dim = [784]
        for neurons in genome['nb_neurons']:
            dim.append(neurons)
        dim.append(10)
        rnd = random.randint(0, 2)
        model = NN(genome['activation'][rnd], dim)
        return model

    def generate(self):
        network = {}
        for key in self.nn_param_choices:
            network[key] = self.nn_param_choices[key]
        network['nb_neurons'] = []
        for i in range(network['nb_layers']):
            network['nb_neurons'].append(self.nn_param_choices['nb_neurons'][i])
        return self.decode(network)

    # metric = accuracy or loss
    @staticmethod
    def best_genome(csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    @staticmethod
    def load_model(models_file):
        return pickle.load(open(models_file))
