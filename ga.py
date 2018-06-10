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
        self.mutate_chance = 0.05
        self.keep = 0.15
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
        self.metric = 'loss'
        self.objective = "min"
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
        acc = []
        for i in range(len(members)):
            # print("\nmodel {}/{} - generation {}/{}\n"
            #       .format(i + 1, len(members), 1, num_generations))
            v, loss = self.evaluate(members[i])
            # v = res[-1]
            # del res
            # print(v)
            fit.append(loss)
            acc.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.objective)
        # print fit
        print("First Generation: best loss: {:0.4f} , best accuracy: {:0.4f}%"
              .format(self.metric_objective(fit), 100.0*max(acc)))
        # Evolve over
        for gen in range(1, num_generations):
            members = []
            # keep 25% of the best
            bests = pop.getBest(int(pop_size * 0.25))
            members += bests
            # print self.evaluate(members[0])
            # members += pop.getWorst(int(pop_size * 0.05))
            worsts = pop.getWorst(int(pop_size * 0.75))
            while len(members) < pop_size:
                index1 = np.random.randint(len(bests))
                index2 = np.random.randint(len(worsts))
                members.append(self.crossover(bests[index1], worsts[index2]))
            # my_randoms = random.sample(xrange(len(worsts)), int(len(worsts)*0.5))
            for i in range(1, len(members)):
                members[i] = self.mutate(members[i], gen)
            # for i in range(len(bests), pop_size):
            #     if 0.5 > random.random():
            #         members[i] = self.mutate(members[i], gen)
            # for i in range(int(pop_size * 0.95)):  # Crossover
            #     members.append(self.crossover(pop.select(), pop.select()))
            # my_randoms = random.sample(xrange(len(members)), int(len(members)*self.mutate_chance))
            # for i in my_randoms:  # Mutation
            #     members[i] = self.mutate(members[i], gen)
            # keep a percentage of the bests
            # members += pop.getBest(int(pop_size * self.keep))
            # # keep part of the worsts
            # # worsts = pop.getWorst(int(pop_size * self.keep))
            # for i in range(int((pop_size -pop_size*self.keep)*0.15)):
            #     members.append(pop.select())
            # current_pop = len(members)
            # # breed to keep the pop size
            # while pop_size > len(members):
            #     index1 = np.random.randint(current_pop)
            #     index2 = np.random.randint(current_pop)
            #     if index1 != index2:
            #         members.append(self.crossover(members[index1], members[index2]))
            # # we decided which network to keep, let's mutate some of them
            # my_randoms = random.sample(xrange(current_pop), int(current_pop*self.mutate_chance))
            # for i in my_randoms:
                # if self.mutate_chance > random.random():
                # members[i] = self.mutate(members[i], gen)
            fit = []
            acc = []
            for i in range(len(members)):
                # print("\nmodel {0}/{1} - generation {2}/{3}:\n"
                #       .format(i + 1, len(members), gen + 1, num_generations))
                v, loss = self.evaluate(members[i])
                # v = res[-1]
                # del res
                fit.append(loss)
                acc.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.objective)
            # print fit
            print("Generation {}: best loss: {:0.4f}"
                  .format(gen + 1, self.metric_objective(fit)))
            if (gen+1) % 100 ==0:
                print("Best accuracy: {:0.4f}%"
                      .format(100.0*max(acc)))
        return self.genome_handler.load_model('best-model.h5')

    def evaluate(self, model):
        loss, accuracy = None, None
        x_part, y_part = self.get_partial_train()
        accuracy, loss = model.validate_batch(x_part, y_part, 64)
        loss /= len(y_part)

        return accuracy, loss

    def crossover(self, model1, model2):
        model = self.genome_handler.generate()
        i = 0
        for param1, param2 in zip(model1.params, model2.params):
            w1, b1 = param1
            w2, b2 = param2
            for column in range(1, w1.shape[0]+1):
                model.params[i][0][column-1:column] = random.choice((w1[column-1:column], w2[column-1:column]))
            for j in range(b1.shape[0]):
                model.params[i][1][j] = random.choice((b1[j], b2[j]))
            i += 1
        return model

    def mutate(self, model, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(model, num_mutations)

    def get_partial_train(self):
        my_randoms = random.sample(xrange(50000), 100)
        x_partial = [self.x_train[i] for i in my_randoms]
        y_partial = [self.y_train[i] for i in my_randoms]
        return x_partial, y_partial

class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        self.random_keep = 0.03
        scores = fitnesses
        # scores = fitnesses - fitnesses.min()
        # if scores.max() > 0:
        #     scores /= scores.max()
        self.min_max = obj is 'max'
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)
        self.combined = [(self.members[i], self.scores[i]) for i in range(len(self.members))]
        self.combined = sorted(self.combined, key=(lambda x: x[1]), reverse=self.min_max)

    def getBest(self, n):
        combined = sorted(self.combined, key=(lambda x: x[1]), reverse=self.min_max)
        # random.shuffle(combined[:n])
        return [x[0] for x in combined[:n]]

    def getWorst(self, n):
        combined = sorted(self.combined, key=(lambda x: x[1]), reverse=not self.min_max)
        # random.shuffle(combined[:n])
        worsts = [x[0] for x in combined[:n]]
        return worsts

    def select(self):
        dart = random.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
        return self.members[0]
