from __future__ import print_function

from genome_handler import GenomeHandler
import numpy as np

from datetime import datetime
import random as rand
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
        Returns:
            keras model: best model found
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
            print("\nmodel {0}/{1} - generation {2}/{3}:\n" \
                  .format(i + 1, len(members), 1, num_generations))
            res = self.evaluate(members[i], epochs)
            v = res[-1]
            del res
            fit.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.objective)
        print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}" \
              .format(self.metric_objective(fit), np.mean(fit), np.std(fit), 1, self.metric))

        # Evolve over
        for gen in range(1, num_generations):
            members = []
            for i in range(int(pop_size * 0.95)):  # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            members += pop.getBest(pop_size - int(pop_size * 0.95))
            for i in range(len(members)):  # Mutation
                members[i].mutate()
            fit = []
            for i in range(len(members)):
                print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                      .format(i + 1, len(members), gen + 1, num_generations))
                res = self.evaluate(members[i], epochs)
                v = res[-1]
                del res
                fit.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.objective)
            print("Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}" \
                  .format(self.metric_objective(fit), np.mean(fit), np.std(fit), gen + 1, self.metric))

        return GenomeHandler.load_model('best-model.h5')

    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        for i in range(epochs):
            accuracy = model.validate(self.x_train, self.y_train)
        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [accuracy]
            writer.writerow(row)

        met = accuracy
        if self.bssf is -1 or self.metric_op(met, self.bssf) and accuracy is not 0:
            try:
                os.remove('best-model.h5')
            except OSError:
                pass
            self.bssf = met
            model.save('best-model.h5')

        return model, accuracy

    def crossover(self, genome1, genome2):
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        return child

    def mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


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
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]