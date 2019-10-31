import random
import string
import datetime
import math

# SETUP
SEED = random.randint(0, 100000)
random.seed(SEED)
print(f'SEED = {SEED}')

GENES = list(string.printable)
# GENES.append(' ')
# GENES.extend(list(string.))
DEBUG = True
POPULATION_SIZE = 1024
MUTATION_RATE = 0.05
MUTATION_INCR = 0.001
EPSILON = 0.001


TARGET = \
'''
I'm the best!
'''

'''
Three Rings for the Elven-kings under the sky,
Seven for the Dwarf-lords in their halls of stone,
Nine for Mortal Men doomed to die,
One for the Dark Lord on his dark throne
In the Land of Mordor where the Shadows lie.
One Ring to rule them all,, One Ring to find them
One Ring to bring them all and in the darkness bind them
In the Land of Mordor where the Shadows lie.
'''


class TextChromosome:
    Target = TARGET

    def __init__(self, genes=None):
        if genes:
            assert len(TextChromosome.Target) == len(genes), f"{genes}"
            self.genes = genes
        else:
            self.genes = random.choices(GENES, k=len(TextChromosome.Target))

        self._score = None

    @property
    def score(self):
        if not self._score:
            assert len(TextChromosome.Target) == len(self.genes), \
                f"{len(TextChromosome.Target)} != {len(self.genes)}"

            self._score = len([1 for i in range(len(TextChromosome.Target))
                               if self.genes[i] == TextChromosome.Target[i]]) / len(TextChromosome.Target)

        return self._score

    def fitness(self):
        return math.exp(self.score)

    def __lt__(self, other):
        return self.score < other.score or \
               self.score == other.score and str(self) < str(other)

    def crossover(self, other):
        assert len(self.genes) == len(other.genes)

        child_genes = []
        for i in range(len(self.genes)):
            r = random.random()

            if r < MUTATION_RATE:
                gene = random.choice(GENES) if random.random() < MUTATION_RATE else self.genes[i]
            elif r < (1.0 - MUTATION_RATE) / 2:
                gene = self.genes[i]
            else:
                gene = other.genes[i]

            child_genes.append(gene)

        return TextChromosome(child_genes)

    def half_crossover(self, other):
        assert len(self.genes) == len(other.genes)

        mid = random.randint(1, len(self.genes)-1)#len(self.genes) // 2
        child_genes = list(map(lambda g: random.choice(GENES) if random.random() < MUTATION_RATE else g,
                               self.genes[:mid] + other.genes[mid:]))
        return TextChromosome(child_genes)

    def __repr__(self):
        return repr(self.genes)

    def __str__(self):
        return ''.join(self.genes)


class Population:
    def __init__(self, population=None):
        if not population:
            self.population = list()
        else:
            self.population = population

    def size(self):
        return len(self.population)

    def add(self, phenotype: TextChromosome):
        self.population.append(phenotype)

    def __repr__(self):
        return repr(self.population)

    def __str__(self):
        return '\n'.join([str(p) for p in self.population])

    def show(self):
        print('-' * 30)
        for phenotype in self.population:
            print(f'{phenotype}  {phenotype.score}')
        print('+' * 30)


def intialize():
    '''
    Generates Population
    '''
    p = Population()
    for i in range(POPULATION_SIZE):
        p.add(TextChromosome())

    return p


def select(p: Population):
    # selected = random.choices(p.population, weights=[ph.score for ph in p.population], k=int(p.size() * d))
    # selected = set(selected)
    selected = sorted(p.population, reverse=True)[:int(p.size()*0.6)]
    selected.extend(random.choices(p.population, weights=[ph.fitness() for ph in p.population], k=int(p.size() * 0.1)))
    return Population(selected)


def new_generation(p: Population):
    while p.size() != POPULATION_SIZE:
        sample = random.sample(p.population, 2)
        x, y = sample[0], sample[-1]

        child = x.crossover(y)
        if DEBUG:
            print('--------------crossover--------------')
            print(f'x = {x}')
            print(f'y = {y}')
            print(f'c = {child}')
            print('-------------------------------------')
        p.add(child)

    return p


def main(log_file):
    global MUTATION_RATE
    best_of_best = 0.0

    if not DEBUG:
        log_file.write('GENERATION,MUTATION_RATE,SCORE,DIFF\n')

    p = intialize()

    for generation in range(1000):
        # print(f'Generation: {i}')

        p = select(p)
        # print('select:')
        # p.show()

        p = new_generation(p)
        # print('new_generation:')
        # p.show()

        # p.population.sort(key=lambda x: x.score, reverse=True)
        best = max(p.population, key=lambda x: x.score)

        if not DEBUG:
            print(f'\rGeneration: {generation}, MUTATION_RATE: {MUTATION_RATE}, SCORE: {best.score}, DIFF: {best.score - best_of_best}',
                end='')

            log_file.write(f'{generation},{MUTATION_RATE:.4},{best.score:.4},{best.score - best_of_best:.4}\n')

        if DEBUG:
            print(f'Generation: {generation}\tMUTATION_RATE: {MUTATION_RATE}\tBEST: {best.score}\tDIFF: {best.score - best_of_best}')
            print(f'{best}')

        if generation % 10 == 0:
            # if abs(best.score - best_of_best) < EPSILON:
            #     MUTATION_RATE = MUTATION_RATE+MUTATION_INCR if MUTATION_RATE < 0.9 else MUTATION_RATE
            # else:
            #     MUTATION_RATE = MUTATION_RATE if MUTATION_RATE <= 0.01 else MUTATION_RATE - MUTATION_INCR
            # # elif best_of_best > best.score:
            # print('-' * 80)
            # print(f'{best}')
            if best.score > best_of_best:
                best_of_best = best.score

        if str(best) == TARGET:
            print(f'DONE: {best.score}\n{best}')
            break

    best = max(p.population, key=lambda x: x.score)
    print(f'LAST: {best.score}\n{best}')

    if not DEBUG:
        log_file.close()


if __name__ == '__main__':
    with open(f'logs/log_{SEED}_{datetime.datetime.today().strftime("%Y-%m-%d-%H.%M.%S")}.txt', 'w') as log_file:
        main(log_file)








