from lib.chromosome import TextChromosome, TextPopulation, GeneticAlgorithm

import time

TARGET = \
'''
Hello World!
'''

target = TextChromosome(list(TARGET))

population = TextPopulation(64, TextChromosome, len(target))

start = time.time()

ga = GeneticAlgorithm(0.1, 0.5, 0.005, population, target)
ga.run(100000)

print(f'Done in {time.time() - start}')
