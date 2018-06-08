# from __future__ import print_function

from genome_handler import GenomeHandler
import numpy as np

from datetime import datetime
import random
import csv
import operator
import os

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class GA:

    def __init__(self, genome_handler, data_path=""):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self.bssf = -1
        random.seed(0)
        # if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
        #     raise ValueError(
        #         'Non-empty file %s already exists. Please change file path to prevent overwritten genome data.' % data_path)
        #
        # print("Genome encoding and accuracy data stored at", self.datafile, "\n")
        # with open(self.datafile, 'a') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     # genome = genome_handler.genome_representation() + ["Val Loss", "Val Accuracy"]
        #     writer.writerow(genome)

    def set_objective(self):
        """set the metric and objective for this search  should be 'accuracy' or 'loss'"""
        self.metric = 'accuracy'
        self.objective = "max"
        self.metric_index = -1
        self.metric_op = METRIC_OPS[self.objective is 'max']
        self.metric_objective = METRIC_OBJECTIVES[self.objective is 'max']

    def run(self, train_set, valid_set, test_set, num_generations, pop_size, epochs, fitness=None, metric='accuracy'):
        """run genetic search on dataset given number of generations and population size
        Args:
            dataset : tuple or list of numpy arrays in form ((train_data, train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs to run each search, passed to keras model.fit -currently searches are
                            curtailed if no improvement is seen in 1 epoch
            fitness (None, optional): scoring function to be applied to population scores, will be called on a numpy array
                                      which is a  min/max scaled version of evaluated model metrics, so
                                      It should accept a real number including 0. If left as default just the min/max
                                      scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what to optimize during search
        """
        self.set_objective()
        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.x_test = test_set[0]
        self.y_test = test_set[1]
        self.x_valid = valid_set[0]
        self.y_valid = valid_set[1]
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        metric_index = 1 if self.metric is 'loss' else -1
        for i in range(len(members)):
            # print("\nmodel {}/{} - generation {}/{}\n"
            #       .format(i + 1, len(members), 1, num_generations))
            v = self.evaluate(members[i], epochs)
            # v = res[-1]
            # del res
            # print(v)
            fit.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.objective)
        print("First Generation: best {}: {:0.4f}% average: {:0.4f}"
              .format(self.metric, 100.0*self.metric_objective(fit), np.mean(fit)))

        # Evolve over
        for gen in range(1, num_generations):
            members = []
            for i in range(int(pop_size * 0.95)):  # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            members += pop.getBest(pop_size - int(pop_size * 0.95))
            for i in range(len(members)):  # Mutation
                members[i] = self.mutate(members[i], gen)
            fit = []
            for i in range(len(members)):
                # print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                #       .format(i + 1, len(members), gen + 1, num_generations))
                v = self.evaluate(members[i], epochs)
                # v = res[-1]
                # del res
                fit.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.objective)
            print("Generation {}: best {}: {:0.4f}% average: {:0.4f}"
                  .format(gen + 1, self.metric, 100.0*self.metric_objective(fit), np.mean(fit)))

        return self.genome_handler.load_model('best-model.h5')

    def evaluate(self, model, epochs):
        loss, accuracy = None, None
        for i in range(epochs):
            x_part, y_part = self.get_partial_train()
            accuracy = model.validate_batch(x_part, y_part, 64)
        # Record the stats
        # with open(self.datafile, 'a') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',',
        #                         quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     row = list(genome) + [accuracy]
        #     writer.writerow(row)

        met = accuracy
        if self.bssf is -1 or self.metric_op(met, self.bssf) and accuracy is not 0:
            try:
                os.remove('best-model.h5')
            except OSError:
                pass
            self.bssf = met
            model.save('best-model.h5')

        return accuracy

    def crossover(self, model1, model2):
        model = random.choice([model1, model2])
        i = 0
        for param1, param2 in zip(model1.params, model2.params):
            w1, _ = param1
            w2, _ = param2
            crossIndexA = random.randint(0, w1.shape[0])
            childW = np.concatenate((w1[:crossIndexA], w2[crossIndexA:]), axis=0)
            model.params[i][0] = childW
            i += 1
        return model

    def mutate(self, model, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(model, num_mutations)

    def get_partial_train(self):
        my_randoms = random.sample(xrange(50000), 128)
        x_partial = [self.x_train[i] for i in my_randoms]
        y_partial = [self.y_train[i] for i in my_randoms]
        return x_partial, y_partial

class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj is 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def getBest(self, n):
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = random.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
