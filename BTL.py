import pygame
import numpy as np
import time
# Mode of the game 
MANUAL_MODE = 0
DFS_MODE = 1
GA_MODE = 2
SCALE_SIZE = 64

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def distance_to(self, p):
		return abs(p.x - self.x) + abs(p.y - self.y)
    
	def __str__(self):
		return str(self.x) + " " + str(self.y)

class Block:
    LIE_HORIZON = 1
    LIE_VERTICAL = 2
    STANDING = 3

    img_standing = pygame.image.load("imgs/block_standing.png")
    img_lying = pygame.image.load("imgs/block_lying.png")

    def __init__(self, status, point_1, point_2):
        self.status = status
        self.point_1 = point_1
        self.point_2 = point_2

    def visualize(self, screen):
        if(self.status == Block.STANDING):
            screen.blit(self.img_standing, (self.point_1.x * SCALE_SIZE, self.point_1.y * SCALE_SIZE))
        else:
            screen.blit(self.img_lying, (self.point_1.x * SCALE_SIZE, self.point_1.y * SCALE_SIZE))
            screen.blit(self.img_lying, (self.point_2.x * SCALE_SIZE, self.point_2.y * SCALE_SIZE))

    def move_up(self):
        if self.status == Block.STANDING:
            self.status = Block.LIE_VERTICAL
            self.point_1.y -= 1
            self.point_2.y -= 2
        elif self.status == Block.LIE_VERTICAL:
            self.status = Block.STANDING
            self.point_1.y -= 2
            self.point_2.y -= 1
        elif self.status == Block.LIE_HORIZON:
            self.point_1.y -= 1
            self.point_2.y -= 1

    def move_down(self):
        if self.status == Block.STANDING:
            self.status = Block.LIE_VERTICAL
            self.point_1.y += 2
            self.point_2.y += 1
        elif self.status == Block.LIE_VERTICAL:
            self.status = Block.STANDING
            self.point_1.y += 1
            self.point_2.y += 2
        elif self.status == Block.LIE_HORIZON:
            self.point_1.y += 1
            self.point_2.y += 1

    def move_right(self):
        if self.status == Block.STANDING:
            self.status = Block.LIE_HORIZON
            self.point_1.x += 1
            self.point_2.x += 2
        elif self.status == Block.LIE_VERTICAL:
            self.point_1.x += 1
            self.point_2.x += 1
        elif self.status == Block.LIE_HORIZON:
            self.status = Block.STANDING
            self.point_1.x += 2
            self.point_2.x += 1

    def move_left(self):
        if self.status == Block.STANDING:
            self.status = Block.LIE_HORIZON
            self.point_1.x -= 2
            self.point_2.x -= 1
        elif self.status == Block.LIE_VERTICAL:
            self.point_1.x -= 1
            self.point_2.x -= 1
        elif self.status == Block.LIE_HORIZON:
            self.status = Block.STANDING
            self.point_1.x -= 1
            self.point_2.x -= 2
    def not_move(self):
        pass
    def encode(self):
        return (str(self))
    def __str__(self):
        return  str(self.status) + " " + str(self.point_1) 

class Map:
    def __init__(self, background, hole, matrix):
        self.background = background
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
        
    def visualize(self, screen):
        pnt = Point(0, 0)
        for i in self.matrix:
            for j in i:
                if j == 1:
                    screen.blit(self.background, (pnt.x * SCALE_SIZE, pnt.y * SCALE_SIZE))
                elif j == 2:
                    screen.blit(self.hole, (pnt.x * SCALE_SIZE, pnt.y * SCALE_SIZE))
                pnt.x += 1
            pnt.x = 0    
            pnt.y += 1
    
    def check_inside_map(self, point):
        return (0 <= point.x and point.x < self.width) and (0 <= point.y and point.y < self.height)

    def check_outof_map(self, block):
        if not self.check_inside_map(block.point_1) or not self.check_inside_map(block.point_2):
            return True
        return False

    def check_outof_ground(self, block):
        point_1_fell = self.matrix[block.point_1.y][block.point_1.x] == 0
        point_2_fell = self.matrix[block.point_2.y][block.point_2.x] == 0
        if point_1_fell or point_2_fell:
            return True
        return False

    def check_win(self, block: Block):
        check_win = self.matrix[block.point_1.y][block.point_1.x] == 2 and block.status == Block.STANDING
        if check_win:
            print("The Block go into hole !!!")
            return True
        return False

    def impact(self, block):
        return self.check_outof_map(block) or self.check_outof_ground(block) or self.check_win(block)


class Start:
    def __init__(self, matrix, init_status, init_point_1, init_point_2):
        self.matrix = matrix
        self.init_status = init_status
        self.init_point_1 = init_point_1
        self.init_point_2 = init_point_2


# stateDemo = State(
#     [
#         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#         [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
#         [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
#     ],
#     Block.STANDING,
#     Point(1, 2),
#     Point(1, 2)
# )
state1 = Start(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    ],
    Block.STANDING,
    Point(1,1),
    Point(1,1)
)

class Movement:

    def __init__(self, map: Map) -> None:
        self.solution = []             # (%2)  
        self.map = map # 0            1            2           3         4=> abs()
        self.moves = ["move_left", "move_right", "move_up", "move_down","not_move"]

    def solve(self, src: Block):
        pass

    def move_to_goal(self, block):
        return self.map.check_win(block)

    def move_out(self, block):
        return self.map.check_outof_map(block) or self.map.check_outof_ground(block)

    @staticmethod
    def print_stack(st):
        for b in st:
            print(b, end=' ')
        print()

import copy
import random

class Individual():
    
    length_of_genes = 30
    @classmethod
    def random_1_gene(self):
        gene = random.choice("0123") 
        return gene

    @classmethod
    def create_1_gene(self,len = 100):
       
        self.length_of_genes = len
        return [self.random_1_gene() for _ in range(self.length_of_genes)]

    def __init__(self, chromosome):
        self.chromosome = chromosome
        
    def cross_and_mutate(self, par2, generation):
        child_chromosome = []
    
        c = random.randint(0,self.length_of_genes) 

        prob = random.random()
        child_chromosome = self.chromosome[:c] + par2.chromosome[c:] 
        mutation_prob = 0.8 if generation > 100 else 0.6 if generation > 50 else 0.4
        if prob < mutation_prob:
            leng_muta = self.length_of_genes
            a = random.randrange(0,leng_muta)
            for _ in range(a):
                p = random.randrange(0,self.length_of_genes)
                child_chromosome[p]= self.random_1_gene()
                
        return Individual(child_chromosome)

class DepthFirstSearch(Movement):

    def __init__(self, map: Map) -> None:
        super().__init__(map)
        self.stack = []
        self.path = dict()
        self.visited = set()

    def solve(self, src : Block) :
        self.stack.append(src)
        self.visited.add(src.encode())
        while len(self.stack) > 0:
            u = self.stack.pop()
            if self.move_to_goal(u):
                solution = []
                while u is not None:
                    solution += [u]
                    if u.encode() in self.path:
                        u = self.path[u.encode()]
                    else:
                        break
                    
                solution.reverse()
                self.print_stack(solution)
                return True, list(solution)

            prev_state = copy.deepcopy(u)
            
            for move_name in self.moves:
                print(move_name, end = "---\n")
                move = getattr(Block, move_name)
                move(u)
                if not self.move_out(u):
                    encode = u.encode()
                    if encode not in self.visited:
                        self.stack.append(u)
                        self.path[encode] = prev_state
                        self.visited.add(encode)
                u = copy.deepcopy(prev_state)

        return False, []

class GeneticAlgorithm(Movement):
    
    def __init__(self, map: Map):
        super().__init__(map)
        self.roof = (self.map.width + self.map.height) * 3
        self.goal = map.goal
    
    def fitness_function(self, ind : Individual):
        block = copy.deepcopy(self.block)
        mt = 0
        
        for i,gen in enumerate(ind.chromosome):
            move_name = self.moves[int(gen)]
            move = getattr(Block, move_name)
            move(block)
            if self.move_out(block):
                b = int(gen)
                move_name = self.moves[b-b%2+(b+1)%2]
                move = getattr(Block,move_name) # 012 (3 2)
                move(block)
                ind.chromosome[i] = '4'
            else:
                mt += (move_name != 'not_move')

            if self.move_to_goal(block):
                ind.chromosome[(i+1):] = '4' * (len(ind.chromosome) - i)
                return self.roof
        # ['' '' '' ] (roof sm(MH) sum(MT)) ['' '' '4...'] heuristic
        # length_of_genes
        point_1_to_goal = self.goal.distance_to(block.point_1) # abs(x-x) + abs(y-y)
        point_2_to_goal = self.goal.distance_to(block.point_2)
        if point_2_to_goal == 0 or point_1_to_goal == 0:
            point_2_to_goal = point_1_to_goal = 2
            
        mh = min(point_1_to_goal,point_2_to_goal) # # #
        fitness = self.roof - mh - mt * 0.1 # mh large - càng xa - fitness càng thấp
        # mh bé, mt bé, mh quan trọng hơn mt
        return fitness
    #dung lm wait
    def select(self, weight):
        r = random.random()
        for i in range(len(weight)):
            if r <= weight[i]:
                return i

    def solve(self, src : Block,size_genetic = 100):
        POPULATION_SIZE = size_genetic

        generation = 1
        found = False
        population = []
        self.block = copy.deepcopy(src)
        
        weight_select = [0]*POPULATION_SIZE # [0,0...]
        weight_select[0] = POPULATION_SIZE//10 #[10, 00000..]
        
        for i in range(1,POPULATION_SIZE):
            weight_select[i]=weight_select[i-1] + POPULATION_SIZE//100 # [10, 11, 12,..., 110]
        
        s = sum(weight_select)
        weight_select = np.array(weight_select)/s # freq
        
        for i in range(1,POPULATION_SIZE):
            weight_select[i] += weight_select[i-1] # [freq, f + 1, f+(f+1),]
        
        for _ in range(POPULATION_SIZE): # ---
            gnome = Individual.create_1_gene(30) # 1 individual
            population.append(Individual(gnome)) #population: list[Individual]

        while not found:
            population = sorted(population, key=lambda x: self.fitness_function(x))
            if self.fitness_function(population[-1]) == self.roof:
                found = True
                break
            new_generation = population[-POPULATION_SIZE*10//100:]

            b = ['L','R','U','D','NONE']
            s = [b[int(i)] for i in population[-1].chromosome]
            print("Generation {}: {}. Fitness: {}".format(
                generation, s, self.fitness_function(population[-1])))
            
            for _ in range(POPULATION_SIZE - POPULATION_SIZE*10//100):
                parent1 = population[self.select(weight_select)]
                parent2 = population[self.select(weight_select)]
                #parent1 cross_and_mutate parent2
                child = parent1.cross_and_mutate(parent2, generation)
                new_generation.append(child)

            population = new_generation
            generation += 1
        
        b = ['L','R','U','D','NONE']
        s = [b[int(i)] for i in population[-1].chromosome]
        print("Generation {}: {}".format(generation, s))
        
        for gen in population[-1].chromosome:
            move_name = self.moves[int(gen)]
            move = getattr(Block, move_name)
            move(src)
            self.solution.append(copy.deepcopy(src))
        
        return True, self.solution

MODE = GA_MODE

# init
pygame.init()

# set title
pygame.display.set_caption("BLOXORZ")
# state of the game
state = state1

# setting map
background_map = pygame.image.load("imgs/map.png")
hole_map = pygame.image.load("imgs/hole.png")
matrix = state.matrix
map = Map(background_map, hole_map, matrix)

# setting block
init_status = state.init_status
init_point_1 = state.init_point_1
init_point_2 = state.init_point_2
block = Block(init_status, copy.copy(init_point_1), copy.copy(init_point_2))

# create the screen
screen = pygame.display.set_mode((map.width * SCALE_SIZE, map.height * SCALE_SIZE))


if MODE == GA_MODE:
# GA set up
    global ga
    ga = GeneticAlgorithm(map)
    global solution1
    _, solution1 = ga.solve(copy.deepcopy(block))
elif MODE != MANUAL_MODE:
    global dfs
    dfs = DepthFirstSearch(map)
    global solution2
    a, solution2 = dfs.solve(copy.deepcopy(block))
# start game
i = 1
running = True
while running:
    
    screen.fill((255, 255, 255))
    map.visualize(screen)
    block.visualize(screen)
    
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
            block.point_1 = copy.copy(init_point_1)
            block.point_2 = copy.copy(init_point_2)

    if MODE == DFS_MODE:
        if i < len(solution):
            block = solution[i]
            i += 1
        time.sleep(0.5)

    if MODE == GA_MODE:
        if i < len(solution2):
            block = solution2[i]
            i += 1
        time.sleep(0.5)


    pygame.display.update()