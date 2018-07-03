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
        self.genome_handler = genome_handler
        random.seed(0)

        self.randomize = np.arange(50000)
        np.random.shuffle(self.randomize)
        self.size_keep = 64

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
        acc = []
        x_part, y_part = self.get_partial_train()
        for i in range(len(members)):
            v, loss = self.evaluate(members[i], x_part, y_part)
            fit.append(loss)
            acc.append(v)

        fit = np.array(fit)
        pop = Population(members, fit, fitness)
        # print fit
        print("First Generation: best loss: {:0.4f} , best accuracy: {:0.4f}%"
              .format(min(fit), 100.0*max(acc)))
        best_loss = min(fit)
        # Evolve over
        elitisme = 1
        prev_loss = 50
        prev_acc = 0
        for gen in range(1, num_generations):
            members = []
            # keep 25% of the best
            bests = pop.get_best(int(pop_size * 0.10))
            members += bests
            worsts = pop.get_worst(int(pop_size * 0.90))
            while len(members) < pop_size:
                index1 = np.random.randint(len(bests))
                index2 = np.random.randint(len(worsts))
                members.append(self.crossover(bests[index1], worsts[index2]))
            for i in range(elitisme, len(members)):
                members[i] = self.genome_handler.mutate(members[i], max(1, gen//4))
            # elitisme = 1
            fit = []
            acc = []
            x_part, y_part = self.get_partial_train()
            for i in range(len(members)):
                v, loss = self.evaluate(members[i], x_part, y_part)
                fit.append(loss)
                acc.append(v)

            fit = np.array(fit)
            pop = Population(members, fit, fitness)

            print("Generation {}: best loss: {:0.4f}, best accuracy: {:0.4f}%".format(gen + 1, min(fit), 100.0*max(acc)))
            if max(acc) >= 0.91:
                np.random.shuffle(self.randomize)
                self.size_keep = min(50000, self.size_keep + 64)
                print self.size_keep, (gen+1)
            if (gen+1) % 100 == 0:
                best_loss = min(fit)
                best_model_acc_valid, best_model_loss_valid = self.evaluate(pop.get_best(1)[0], self.x_valid, self.y_valid)
                best_model_acc, best_model_loss = self.evaluate(pop.get_best(1)[0], self.x_train, self.y_train)
                if best_model_loss < prev_loss:
                    pop.get_best(1)[0].save("model_from_ga.model")
                else:
                    np.random.shuffle(self.randomize)
                prev_loss = best_model_loss
                prev_acc = best_model_acc
                print("Best loss on valid: {:0.4f}, best accuracy on valid: {:0.4f}%".format(best_model_loss_valid, 100.0*best_model_acc_valid))
                print("Best loss on test: {:0.4f}, best accuracy on test: {:0.4f}%".format(best_model_loss, 100.0*best_model_acc))
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
        x_partial = [self.x_train[i] for i in self.randomize[:self.size_keep]]
        y_partial = [self.y_train[i] for i in self.randomize[:self.size_keep]]
        return x_partial, y_partial


class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score):
        self.members = members
        self.random_keep = 0.03
        scores = fitnesses
        self.min_max = False
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
