import pygame
import numpy as np


# Mode of the game 
MANUAL_MODE = 0
DFS_MODE = 1
GA_MODE = 2
FITNESS_DISTANCE_ZERO = 1e7

# size on screen
CELL_SIZE = 64

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def distance_to(self, p):
		return abs(p.x - self.x) + abs(p.y - self.y)

    
	def __str__(self):
		return str(self.x) + " " + str(self.y)

class Block:

    # status of block
    LYING_HORIZONTALLY = 1
    LYING_VERTICALLY = 2
    STANDING = 3

    # images are showed on screen
    standing = pygame.image.load("imgs/block_standing.png")
    lying = pygame.image.load("imgs/block_lying.png")

    def __init__(self, status, fst_point, snd_point):
        self.status = status
        self.fst_point = fst_point
        self.snd_point = snd_point

    def draw(self, screen):
        if(self.status == Block.STANDING):
            screen.blit(self.standing, (self.fst_point.x * CELL_SIZE, self.fst_point.y * CELL_SIZE))
        else:
            screen.blit(self.lying, (self.fst_point.x * CELL_SIZE, self.fst_point.y * CELL_SIZE))
            screen.blit(self.lying, (self.snd_point.x * CELL_SIZE, self.snd_point.y * CELL_SIZE))

    def move_up(self):
        if self.status == Block.STANDING:
            self.status = Block.LYING_VERTICALLY
            self.fst_point.y -= 1
            self.snd_point.y -= 2
        elif self.status == Block.LYING_VERTICALLY:
            self.status = Block.STANDING
            self.fst_point.y -= 2
            self.snd_point.y -= 1
        elif self.status == Block.LYING_HORIZONTALLY:
            self.fst_point.y -= 1
            self.snd_point.y -= 1

    def move_down(self):
        if self.status == Block.STANDING:
            self.status = Block.LYING_VERTICALLY
            self.fst_point.y += 2
            self.snd_point.y += 1
        elif self.status == Block.LYING_VERTICALLY:
            self.status = Block.STANDING
            self.fst_point.y += 1
            self.snd_point.y += 2
        elif self.status == Block.LYING_HORIZONTALLY:
            self.fst_point.y += 1
            self.snd_point.y += 1

    def move_right(self):
        if self.status == Block.STANDING:
            self.status = Block.LYING_HORIZONTALLY
            self.fst_point.x += 1
            self.snd_point.x += 2
        elif self.status == Block.LYING_VERTICALLY:
            self.fst_point.x += 1
            self.snd_point.x += 1
        elif self.status == Block.LYING_HORIZONTALLY:
            self.status = Block.STANDING
            self.fst_point.x += 2
            self.snd_point.x += 1

    def move_left(self):
        if self.status == Block.STANDING:
            self.status = Block.LYING_HORIZONTALLY
            self.fst_point.x -= 2
            self.snd_point.x -= 1
        elif self.status == Block.LYING_VERTICALLY:
            self.fst_point.x -= 1
            self.snd_point.x -= 1
        elif self.status == Block.LYING_HORIZONTALLY:
            self.status = Block.STANDING
            self.fst_point.x -= 1
            self.snd_point.x -= 2
    
    def encode(self):
        return (str(self))
    
    def __str__(self):
        return  str(self.status) + " " + str(self.fst_point) 

class Map:

    def __init__(self, cell, hole, matrix):
        self.cell = cell
        self.hole = hole
        self.matrix = matrix
        self.height = len(matrix)
        self.width = len(matrix[0])

    def draw(self, screen):
        pnt = Point(0, 0)
        for i in self.matrix:
            for j in i:
                if j == 1:
                    screen.blit(self.cell, (pnt.x * CELL_SIZE, pnt.y * CELL_SIZE))
                elif j == 2:
                    screen.blit(self.hole, (pnt.x * CELL_SIZE, pnt.y * CELL_SIZE))
                pnt.x += 1
            pnt.x = 0    
            pnt.y += 1
    
    def point_in_map(self, point):
        return (0 <= point.x and point.x < self.width) and (0 <= point.y and point.y < self.height)

    def block_out_map(self, block):
        if not self.point_in_map(block.fst_point) or not self.point_in_map(block.snd_point):
            print("Your Block Got Out The Map !!!")
            return True
        return False

    def block_felt(self, block):
        fst_point_fell = self.matrix[block.fst_point.y][block.fst_point.x] == 0
        snd_point_fell = self.matrix[block.snd_point.y][block.snd_point.x] == 0
        if fst_point_fell or snd_point_fell:
            print("Your Block Fell !!!")
            return True
        return False

    def won_the_game(self, block: Block):
        won_the_game = self.matrix[block.fst_point.y][block.fst_point.x] == 2 and block.status == Block.STANDING
        if won_the_game:
            print("You Won <--(^_^)-->")
            return True
        return False

    def impact(self, block):
        return self.block_out_map(block) or self.block_felt(block) or self.won_the_game(block)

#    def hole_point(self):

class State:
    def __init__(self, matrix, init_status, init_fst_point, init_snd_point):
        self.matrix = matrix
        self.init_status = init_status
        self.init_fst_point = init_fst_point
        self.init_snd_point = init_snd_point


stateDemo = State(
    [
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    ],
    Block.STANDING,
    Point(1, 2),
    Point(1, 2)
)

state1 = State(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    ],
    Block.STANDING,
    Point(1,1),
    Point(1,1)
)

class Solution:

    def __init__(self, map: Map) -> None:
        self.solution = []
        self.map = map
        self.moves = ["move_left", "move_right", "move_up", "move_down"]

    def solve(self, src: Block):
        pass

    def is_goal(self, block):
        return self.map.won_the_game(block)

    def is_failed(self, block):
        return self.map.block_out_map(block) or self.map.block_felt(block)

    @staticmethod
    def print_stack(st):
        for b in st:
            print(b, end=' ')
        print()

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
        gnome_len = 30
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
        '''for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            # self: parent1 1 invididual 1 string
            # par2 1 string    [("1","0"),("2","3"),......,("2","1")]
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
                child_chromosome.append(self.mutated_genes()) # '''
        c = random.randint(0,len(self.chromosome)) 
        # create new Individual(offspring) using
        # generated chromosome for offspring
        prob = random.random()
        child_chromosome = self.chromosome[:c] + par2.chromosome[c:] 
        if prob < 0.7:
            c = random.randrange(1,len(child_chromosome))
            child_chromosome = child_chromosome[:(c-1)] + [self.mutated_genes()] + child_chromosome[c:] 
        
        return Individual(child_chromosome)



class GeneticAlgorithm(Solution):
    
    def __init__(self, map: Map):
        super().__init__(map)

    def cal_fitness(self, ind : Individual):
        block = copy.deepcopy(self.block)
        for gen in ind.chromosome:
            move_name = self.moves[int(gen)]
            # print(move_name, end="---\n")
            move = getattr(Block, move_name)
            move(block)
            if self.is_failed(block):
                break
            if self.is_goal(block):
                return 3
        
        hole = Point(4,7) # 
        f_distance = hole.distance_to(block.fst_point)
        s_distance = hole.distance_to(block.fst_point)
        if s_distance == 0:
            s_distance = 2
        fitness = 1/min(f_distance,s_distance)

        return fitness
    def solve1(self, populations):
        weight_selection = [self.cal_fitness(population) for population in populations]
        for i in range(1,len(weight_selection)):
            weight_selection[i] += weight_selection[i-1]
        
        pass
    def solve(self, src : Block,size_genetic = 100):
        POPULATION_SIZE = size_genetic

        #current generation
        generation = 1
        found = False
        population = []
        self.block = copy.deepcopy(src)
        # create initial population

        for _ in range(POPULATION_SIZE):
            gnome = Individual.create_gnome() # 1 individual
            population.append(Individual(gnome)) #population: list[Individual]
        
        while not found:
            '''
            population = [Indi]
            scores = [fitness_score]
            scores_pop = zip(scores,population)
            if find_index(0,scores) == -1 
            => found => end
            else: loop
            
            '''
            '''
            # sort the population in increasing order of fitness score
            population = np.argpartition(population, lambda x: self.cal_fitness(x, copy.deepcopy(src)),)
            a = self.population_fitness(population)
            # if the individual having lowest fitness score ie.
            # 0 then we know that we have reached to the target
            # and break the loop
            if (population[np.argmax(population)]):
                found = True
                break
            '''
            # sort the population in increasing order of fitness score
            population = sorted(population, key=lambda x: self.cal_fitness(x))
            
            # if the individual having lowest fitness score ie.
            # 0 then we know that we have reached to the target
            # and break the loop
            print(self.cal_fitness(population[-1]))
            if self.cal_fitness(population[-1]) == 3:
                found = True
                break
            # Otherwise generate new offsprings for new generation
            new_generation = [] if POPULATION_SIZE%2 == 0 else [population[-1],population[-2]]
            # new_generation.extend(population[s:]) # 

            # From 70% of fittest population, Individuals
            # will mate to produce offspring
            for _ in range(POPULATION_SIZE//2):
                parent1 = random.choice(population[7*POPULATION_SIZE//10:])
                parent2 = random.choice(population[7*POPULATION_SIZE//10:])
                #parent1 mate parent2

                child1 = parent1.mate(parent2)
                child2 = parent2.mate(parent1)

                new_generation.append(child1)
                new_generation.append(child2)

            population = new_generation

            print("Generation {}: {}".format(
                generation, population[-1].chromosome))

            generation += 1
        b = ['L','R','U','D']
        s = [b[int(i)] for i in population[-1].chromosome]
        print("Generation {}: {}".format(
                generation, s))
        for gen in population[-1].chromosome:
            move_name = self.moves[int(gen)]
            # print(move_name, end="---\n")
            move = getattr(Block, move_name)
            move(src)
            self.solution.append(copy.deepcopy(src))
        
        return True, self.solution

import time
MODE = GA_MODE

# init
pygame.init()

# set title
pygame.display.set_caption("BLOXORZ")
icon = pygame.image.load("imgs/icon.png")
pygame.display.set_icon(icon)

# state of the game
state = state1

# setting map
cell_map = pygame.image.load("imgs/map.png")
hole_map = pygame.image.load("imgs/hole.png")
matrix = state.matrix
map = Map(cell_map, hole_map, matrix)

# setting block
init_status = state.init_status
init_fst_point = state.init_fst_point
init_snd_point = state.init_snd_point
block = Block(init_status, copy.copy(init_fst_point), copy.copy(init_snd_point))

# create the screen
screen = pygame.display.set_mode((map.width * CELL_SIZE, map.height * CELL_SIZE))



# GA set up
ga = GeneticAlgorithm(map)
_, solution2 = ga.solve(copy.deepcopy(block))

# start game
i = 1
running = True
while running:
    
    screen.fill((255, 255, 255))
    map.draw(screen)
    block.draw(screen)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if MODE == MANUAL_MODE:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    block.move_left()
                if event.key == pygame.K_RIGHT:
                    block.move_right()
                if event.key == pygame.K_UP:
                    block.move_up()
                if event.key == pygame.K_DOWN:
                    block.move_down()   

    if MODE == MANUAL_MODE:
        if map.impact(block) == True:
            block.status = init_status
            block.fst_point = copy.copy(init_fst_point)
            block.snd_point = copy.copy(init_snd_point)

    # if MODE == DFS_MODE:
    #     if i < len(solution):
    #         block = solution[i]
    #         i += 1
    #     time.sleep(0.5)

    if MODE == GA_MODE:
        if i < len(solution2):
            block = solution2[i]
            i += 1
        time.sleep(0.5)


    pygame.display.update()