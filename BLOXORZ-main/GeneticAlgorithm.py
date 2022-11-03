from Point import Point
from Solution import Solution
from Map import Map
from Block import Block
import copy
import random

class Individual(object):
    
    GENS = "0123"

    @classmethod
    def mutated_genes(self):
        '''
        create random genes for mutation
        '''
        gene = random.choice(self.GENS)
        return gene

    @classmethod
    def create_gnome(self):
        '''
        create chromosome or string of genes
        '''
        gnome_len = 100
        return [self.mutated_genes() for _ in range(gnome_len)]

    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''

        # chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):

            # random probability
            prob = random.random()

            # if prob is less than 0.45, insert gene
            # from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)

            # if prob is between 0.45 and 0.90, insert
            # gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)

            # otherwise insert random gene(mutate),
            # for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())

        # create new Individual(offspring) using
        # generated chromosome for offspring
        return Individual(child_chromosome)



class GeneticAlgorithm(Solution):
    
    def __init__(self, map: Map):
        super().__init__(map)

    def cal_fitness(self, ind : Individual, block : Block):

        for gen in ind.chromosome:
            move_name = self.moves[int(gen)]
            # print(move_name, end="---\n")
            move = getattr(Block, move_name)
            move(block)
            if self.is_failed(block):
                break
            if self.is_goal(block):
                return 0
        
        hole = Point(4,7)
        fitness = min(hole.distance_to(block.fst_point), hole.distance_to(block.snd_point))
        return fitness

    def solve(self, src : Block):
        POPULATION_SIZE = 100

        #current generation
        generation = 1

        found = False
        population = []

        # create initial population
        for _ in range(POPULATION_SIZE):
            gnome = Individual.create_gnome()
            population.append(Individual(gnome))

        while not found:

            # sort the population in increasing order of fitness score
            population = sorted(population, key=lambda x: self.cal_fitness(x, copy.deepcopy(src)))

            # if the individual having lowest fitness score ie.
            # 0 then we know that we have reached to the target
            # and break the loop
            if self.cal_fitness(population[0], copy.deepcopy(src)) == 0:
                found = True
                break

            # Otherwise generate new offsprings for new generation
            new_generation = []

            # Perform Elitism, that mean 10% of fittest population
            # goes to the next generation
            s = int((10*POPULATION_SIZE)/100)
            new_generation.extend(population[:s])

            # From 50% of fittest population, Individuals
            # will mate to produce offspring
            s = int((90*POPULATION_SIZE)/100)
            for _ in range(s):
                parent1 = random.choice(population[:50])
                parent2 = random.choice(population[:50])
                child = parent1.mate(parent2)
                new_generation.append(child)

            population = new_generation

            print("Generation {}: {}".format(
                generation, population[0].chromosome))

            generation += 1

        print("Generation {}: {}".format(
                generation, population[0].chromosome))

        for gen in population[0].chromosome:
            move_name = self.moves[int(gen)]
            # print(move_name, end="---\n")
            move = getattr(Block, move_name)
            move(src)
            self.solution.append(src)
        
        return True, self.solution
