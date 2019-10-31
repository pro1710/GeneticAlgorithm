import abc
import string
import random
import math

class BaseIndividual(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def create_gnome(cls, length):
        pass

    @property
    @abc.abstractmethod
    def genes(self):
        pass

    @abc.abstractmethod
    def compare(self, other):
        pass

    @property
    @abc.abstractmethod
    def fitness(self):
        pass

    @classmethod
    @abc.abstractmethod
    def crossover(cls, x, y, mutation_rate):
        pass

    @property
    @abc.abstractmethod
    def score(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def display(self):
        pass


class TextChromosome(BaseIndividual):
    TextGenes = list(string.printable)

    def __init__(self, genes):
        self._genes = genes
        self._score = 0

    @classmethod
    def create_gnome(cls, length):
        genes = random.choices(TextChromosome.TextGenes, k=length)
        return cls(genes)

    @property
    def genes(self):
        return self._genes

    def __len__(self):
        return len(self.genes)

    def compare(self, other):
        assert len(self.genes) == len(other.genes)

        res = 0
        for i in range(len(self.genes)):
            if self.genes[i] == other.genes[i]:
                res += 1

        self._score = res / len(self)

        return res

    @property
    def score(self):
        return self._score

    @property
    def fitness(self):
        return self.score ** 2

    @classmethod
    def crossover(cls, x, y, mutation_rate):
        assert len(x.genes) == len(y.genes)

        child_genes = []
        for i in range(len(x)):
            r = random.random()

            if r < mutation_rate:
                gene = random.choice(TextChromosome.TextGenes)
            elif r < (1.0 + mutation_rate) / 2:
                gene = x.genes[i]
            else:
                gene = y.genes[i]

            child_genes.append(gene)

        return cls(child_genes)

    def __str__(self):
        return ''.join(self.genes)

    def display(self):
        print('-'*64)
        print(f'score = {self.score}\n{str(self)}')
        print('-' * 64)


class BasePopulation(abc.ABC):
    @abc.abstractmethod
    def evolve(self, selection_rate, mutation_rate, target: BaseIndividual):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def display(self):
        pass


class TextPopulation(BasePopulation):
    def __init__(self, p_size: BasePopulation, chromosome_cls: TextChromosome, c_len):
        self._population = [chromosome_cls.create_gnome(c_len) for _ in range(p_size)]
        self._alpha = None

    def evolve(self, elite_rate, selection_rate, mutation_rate, target: BaseIndividual):
        self._population.sort(key=lambda ind: ind.fitness, reverse=True)

        elite_num = int(len(self) * elite_rate)

        selection_num = int(len(self) * selection_rate)

        new_generation = self._population[:elite_num]

        while len(new_generation) != len(self._population):
            sample = random.sample(self._population[:selection_num], 2)
            x, y = sample[0], sample[-1]
            new_generation.append(type(target).crossover(x, y, mutation_rate))
        #
        # if random.random() < 0.01:
        #     new_generation.append(TextChromosome.create_gnome(len(new_generation[-1])))

        self._population = new_generation

    def evaluate(self, target: TextChromosome):
        for ind in self._population:
            if not ind.score:
                ind.compare(target)

            if not self._alpha or ind.score > self._alpha.score:
                self._alpha = ind

    @property
    def alpha(self):
        return self._alpha

    def __len__(self):
        return len(self._population)

    def display(self):
        print('-' * 64)
        for ind in self._population:
            print(str(ind))
            print('-'*64)


class GeneticAlgorithm:
    def __init__(self, elite_rate, selection_rate, mutation_rate, population: BasePopulation, target: BaseIndividual):
        self.elite_rate = elite_rate
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.population = population
        self.target = target

    def run_epoch(self):
        self.population.evaluate(self.target)

        self.population.evolve(self.elite_rate, self.selection_rate, self.mutation_rate, self.target)

        return self.population.alpha

    def run(self, generation_limits):
        for generation in range(generation_limits):

            alpha = self.population.alpha
            score = 0.0 if not alpha else alpha.score
            print(f'\rEPOCH: {generation}, SCORE = {score:.4}, POPULATION = {len(self.population)}', end=' ')

            alpha = self.run_epoch()

            # print(f'\r{str(alpha)}', end = '')

            if alpha.score >= 1.0:

                print(f'\nDONE: epoch {generation}')
                alpha.display()
                break

        print(f'\nLAST: limit = {generation_limits}')
        alpha.display()

        print(f'elite_rate={self.elite_rate}, selection_rate={self.selection_rate}, mutation_rate={self.mutation_rate}')







