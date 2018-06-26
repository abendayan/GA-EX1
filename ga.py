import numpy as np
import random
import operator

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class GA:

    def __init__(self, genome_handler):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None
        self.metric = 'loss'
        self.objective = "min"
        self.metric_index = -1
        self.metric_op = METRIC_OPS[self.objective is 'max']
        self.metric_objective = METRIC_OBJECTIVES[self.objective is 'max']
        self.genome_handler = genome_handler
        self.bssf = -1
        self.mutate_chance = 0.05
        self.keep = 0.15
        random.seed(0)
        self.my_randoms = random.sample(xrange(50000), 50)

    def run(self, train_set, valid_set, test_set, num_generations, pop_size, fitness=None):

        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.x_test = test_set[0]
        self.y_test = test_set[1]
        self.x_valid = valid_set[0]
        self.y_valid = valid_set[1]
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        # metric_index = 1 if self.metric is 'loss' else -1
        acc = []
        x_part, y_part = self.get_partial_train()
        for i in range(len(members)):
            v, loss = self.evaluate(members[i], x_part, y_part)
            fit.append(loss)
            acc.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.objective)
        # print fit
        print("First Generation: best loss: {:0.4f} , best accuracy: {:0.4f}%"
              .format(self.metric_objective(fit), 100.0*max(acc)))
        best_loss = min(fit)
        # Evolve over
        for gen in range(1, num_generations):
            members = []
            # keep 25% of the best
            bests = pop.get_best(int(pop_size * 0.25))
            members += bests
            worsts = pop.get_worst(int(pop_size * 0.75))
            while len(members) < pop_size:
                index1 = np.random.randint(len(bests))
                index2 = np.random.randint(len(worsts))
                members.append(self.crossover(bests[index1], worsts[index2]))
            for i in range(5, len(members)):
                members[i] = self.genome_handler.mutate(members[i])
            fit = []
            acc = []
            x_part, y_part = self.get_partial_train()
            for i in range(len(members)):
                v, loss = self.evaluate(members[i], x_part, y_part)
                fit.append(loss)
                acc.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.objective)
            print("Generation {}: best loss: {:0.4f}, best accuracy: {:0.4f}%"
                  .format(gen + 1, self.metric_objective(fit), 100.0*max(acc)))
            if (gen+1) % 100 == 0:
                if min(fit) < best_loss:
                    self.my_randoms += random.sample(xrange(50000), 50)
                    self.my_randoms = list(set(self.my_randoms))
                best_loss = min(fit)
                print len(self.my_randoms)
                print("Best accuracy: {:0.4f}%"
                      .format(100.0*max(acc)))
        best = pop.get_best(1)[0]
        best.save("model_from_ga.model")

    @staticmethod
    def evaluate(model, x_part, y_part):
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

    def get_partial_train(self):
        # my_randoms = random.sample(xrange(50000), self.partial)
        x_partial = [self.x_train[i] for i in self.my_randoms]
        y_partial = [self.y_train[i] for i in self.my_randoms]
        return x_partial, y_partial


class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        self.random_keep = 0.03
        scores = fitnesses
        self.min_max = obj is 'max'
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)
        self.combined = [(self.members[i], self.scores[i]) for i in range(len(self.members))]
        self.combined = sorted(self.combined, key=(lambda x: x[1]), reverse=self.min_max)

    def get_best(self, n):
        combined = sorted(self.combined, key=(lambda x: x[1]), reverse=self.min_max)
        return [x[0] for x in combined[:n]]

    def get_worst(self, n):
        combined = sorted(self.combined, key=(lambda x: x[1]), reverse=not self.min_max)
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
