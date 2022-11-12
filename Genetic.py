from Movement import *
from Individual import *
import copy
import numpy as np

class GeneticAlgorithm(Movement):
    
    def __init__(self, map: Map):
        super().__init__(map)
        self.roof = (self.map.width + self.map.height) * 3
        self.goal = map.goal
    def cal_fitness(self, ind : Individual):
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
        # gen_len
        f_distance = self.goal.distance_to(block.point_1) # abs(x-x) + abs(y-y)
        s_distance = self.goal.distance_to(block.point_2)
        if s_distance == 0 or f_distance == 0:
            s_distance = f_distance = 2
            
        mh = min(f_distance,s_distance) # # #
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
        
        self.block = copy.deepcopy(src)
        #weight
        weight_select = [0]*POPULATION_SIZE # [0,0...]
        weight_select[0] = POPULATION_SIZE//10 #[10, 00000..]
        for i in range(1,POPULATION_SIZE):
            weight_select[i]=weight_select[i-1] + POPULATION_SIZE//100 # [10, 11, 12,..., 110]
        s = sum(weight_select)
        weight_select = np.array(weight_select)/s # freq

        for i in range(1,POPULATION_SIZE):
            weight_select[i] += weight_select[i-1] # [freq, f + 1, f+(f+1),]
        
        population = []
        for _ in range(POPULATION_SIZE): # ---
            gnome = Individual.create_1_gene(30) # 1 individual
            population.append(Individual(gnome)) #population: list[Individual]

        while not found:
            population = sorted(population, key=lambda x: self.cal_fitness(x))
            if self.cal_fitness(population[-1]) == self.roof:
                found = True
                break
            #lấy 10% fitness cao nhất
            new_generation = population[-POPULATION_SIZE*10//100:]

            b = ['L','R','U','D','NONE']
            s = [b[int(i)] for i in population[-1].chromosome]
            print("Generation {}: {}. Fitness: {}".format(
                generation, s, self.cal_fitness(population[-1])))
            
            for _ in range(POPULATION_SIZE - POPULATION_SIZE*10//100):
                parent1 = population[self.select(weight_select)]
                parent2 = population[self.select(weight_select)]
                #parent1 mate parent2
                child = parent1.cross_and_mutate(parent2, generation)
                new_generation.append(child)

            population = new_generation

            

            generation += 1
        
        b = ['L','R','U','D','NONE']
        s = [b[int(i)] for i in population[-1].chromosome]
        print("Generation {}: {}".format(
                generation, s))
        for gen in population[-1].chromosome:
            move_name = self.moves[int(gen)]
            move = getattr(Block, move_name)
            move(src)
            self.solution.append(copy.deepcopy(src))
        
        return True, self.solution