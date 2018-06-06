import numpy as np
import random
from nn import NN as NN


class GenomeHandler:
    def __init__(self, nn_param_choices):
        self.nn_param_choices = nn_param_choices
        self.network = {}

    def mutate(self):
        mutation = random.choice(list(self.nn_param_choices.keys()))
        # Mutate one of the params.
        self.network[mutation] = random.choice(self.nn_param_choices[mutation])

    def decode(self, genome):
        model = NN
        model.activation_function = genome['activation']
        model.hidden_dim = genome['nb_neurons']
        model.layer_size = genome['nb_layers']
        model.batch = genome['batch']
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
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])
        return self.network

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

    def load_model(self, models_file=""):
        pass