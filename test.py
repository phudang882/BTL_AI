import pygame
import numpy as np


# Mode of the game 
MANUAL_MODE = 0
DFS_MODE = 1
GA_MODE = 2
FITNESS_DISTANCE_ZERO = 1e7
# size on screen alo ko nghe ko thấy cái loa đâu hết de tui tao meet
# https://meet.google.com/bmb-zhsj-xjk t yêu cầu rồi đó
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
    def not_move(self):
        pass
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
        self.find_goal()
        
    def find_goal(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i][j] == 2:
                    self.goal = Point(i,j)
        
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
            # print("Your Block Got Out The Map !!!")
            return True
        return False

    def block_felt(self, block):
        fst_point_fell = self.matrix[block.fst_point.y][block.fst_point.x] == 0
        snd_point_fell = self.matrix[block.snd_point.y][block.snd_point.x] == 0
        if fst_point_fell or snd_point_fell:
            # print("Your Block Fell !!!")
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
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0]
    ],
    Block.STANDING,
    Point(1,1),
    Point(1,1)
)

class Solution:

    def __init__(self, map: Map) -> None:
        self.solution = []             # (%2)  
        self.map = map # 0            1            2           3         4=> abs()
        self.moves = ["move_left", "move_right", "move_up", "move_down","not_move"]

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
    
    GENS = "01234"
    gen_len = 30
    @classmethod
    def mutated_genes(self):
        '''
        create random genes for mutation 
        '''
        gene = random.choice(self.GENS) 
        return gene

    @classmethod
    def create_gnome(self,len = 100):
        '''
        create chromosome or string of genes
        '''
        self.gen_len = len
        return [self.mutated_genes() for _ in range(self.gen_len)]

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
        # par2 = Individual("12312")
        #test par2

        c = random.randint(0,self.gen_len) 
        # create new Individual(offspring) using
        # generated chromosome for offspring
        prob = random.random()
        child_chromosome = self.chromosome[:c] + par2.chromosome[c:] 
        if prob < 0.5:
            leng_muta = self.gen_len//4
            a = random.randrange(0,leng_muta)
            for _ in range(a):
                p = random.randrange(0,self.gen_len)
                child_chromosome[p]= self.mutated_genes()
                
        return Individual(child_chromosome)


class GeneticAlgorithm(Solution):
    
    def __init__(self, map: Map):
        super().__init__(map)
        self.roof = (self.map.width + self.map.height) * 3
        self.goal = map.goal
    def cal_fitness(self, ind : Individual):
        block = copy.deepcopy(self.block)
        mt = 0
        
        for i,gen in enumerate(ind.chromosome):
            move_name = self.moves[int(gen)]
            # print(move_name, end="---\n")
            move = getattr(Block, move_name)
            move(block)
            if self.is_failed(block):
                b = int(gen)
                move_name = self.moves[b-b%2+(b+1)%2]
                move = getattr(Block,move_name) # 012 (3 2)
                move(block)
                ind.chromosome[i] = '4'
            else:
                mt += (move_name != 'not_move')

            if self.is_goal(block):
                return self.roof
        # ['' '' '' ] (roof sm(MH) sum(MT)) ['' '' '4...'] heuristic
        # gen_len
        f_distance = self.goal.distance_to(block.fst_point) # abs(x-x) + abs(y-y)
        s_distance = self.goal.distance_to(block.snd_point)
        if s_distance == 0 or f_distance == 0:
            s_distance = f_distance = 2
            
        mh = min(f_distance,s_distance) # # #
        fitness = self.roof - mh - mt * 0.1 # mh large - càng xa - fitness càng thấp
        # mh bé, mt bé, mh quan trọng hơn
        return fitness
    #dung lm wait
    def solve1(self, populations):
        weight_selection = 1
        for i in range(1,len(weight_selection)):
            weight_selection[i] += weight_selection[i-1]
        
        pass
    def select(self, weight):
        r = random.random()
        for i in range(len(weight)):
            if r <= weight[i]:
                return i
        pass
    def solve(self, src : Block,size_genetic = 100):
        POPULATION_SIZE = size_genetic
        self.size_genetic = size_genetic
        #current generation
        generation = 1
        found = False
        population = []
        self.block = copy.deepcopy(src)
        # create initial population
        # while len(population) < POPULATION_SIZE
        weight_select = [0]*POPULATION_SIZE # [0,0...]
        weight_select[0] = POPULATION_SIZE//10 #[10, 00000..]
        for i in range(1,POPULATION_SIZE):
            weight_select[i]=weight_select[i-1] + POPULATION_SIZE//100 # [10, 11, 12,..., 110]
        s = sum(weight_select)
        weight_select = np.array(weight_select)/s # freq
        for i in range(1,POPULATION_SIZE):
            weight_select[i] += weight_select[i-1] # [freq, f + 1, f+(f+1),]
        
        for _ in range(POPULATION_SIZE): # ---
            gnome = Individual.create_gnome(30) # 1 individual
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
            # print(self.cal_fitness(population[-1]))
            if self.cal_fitness(population[-1]) == self.roof:
                found = True
                break
            # choose 30% max
            new_generation = population[-POPULATION_SIZE*10//100:]
            # new_generation.extend(population[s:]) # 

            # From 70% of fittest population, Individuals
            # will mate to produce offspring
            print("Generation {}: {}. Fitness: {}".format(
                generation, population[-1].chromosome, self.cal_fitness(population[-1])))
            
            for _ in range(POPULATION_SIZE - POPULATION_SIZE*10//100):
                parent1 = population[self.select(weight_select)]
                parent2 = population[self.select(weight_select)]
                #parent1 mate parent2
                child = parent1.mate(parent2)
                new_generation.append(child)

            population = new_generation

            

            generation += 1
        
        b = ['L','R','U','D','NONE']
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
# def read_map():
#     with open('map.txt','r') as f:
#         while f.
#     pass
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
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from collections import Counter
# POPULATION_SIZE = 100
# wight_select = [0]*POPULATION_SIZE
# wight_select[0] = POPULATION_SIZE//10
# for i in range(1,POPULATION_SIZE):
#     wight_select[i]=wight_select[i-1] + POPULATION_SIZE//100
# s = sum(wight_select)
# wight_select = np.array(wight_select)/s

# for i in range(1,POPULATION_SIZE):
#     wight_select[i] += wight_select[i-1]

# result = []
# for i in range(POPULATION_SIZE):
#     a = random.random()
#     for j in range(POPULATION_SIZE):
#         if (a <= wight_select[j]):
#             result.append((i,j))
#             break

# result1 = [i for _,i in result]
# print(result1)
# result2 = {}
# for i in result1:
#     x = POPULATION_SIZE//100
#     a = 10*x if i<10*x else 30*x if i<30*x else 50*x if i<50*x else 70*x if i<70*x else 90*x
#     if result2.get(a) is None:
#         result2[a] = 0
#     result2[a]+=1
# plt.pie(result2.values(),labels=result2.keys())
# plt.show()