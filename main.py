from Genetic import *
from DFS import *
import Point
import Start
import pygame
import visualize
def read_to_start(path):
    file = open(path,'r')
    l = [[num for num in line.split()] for line in file]
    l[1:] = [[int(j) for j in i] for i in l[1:]]
    # print(l[1][0])
    start = Start.Start(l[3:],
            init_point_1=Point.Point(l[1][0],l[1][1]),
            init_point_2=Point.Point(l[2][0],l[2][1]),mode=l[0][0])
    return start

start = read_to_start('test.txt')

background_map = pygame.image.load("imgs/map.png")
hole_map = pygame.image.load("imgs/hole.png")
map = Map(background_map,hole_map,start.matrix)
solver = None
solution = None
print(start.status)
print(start.init_point_1 == start.init_point_2)
src = Block(start.status,start.init_point_1,start.init_point_2)
print(start.mode[0])
if start.mode == 'G':
    solver = GeneticAlgorithm(map)
    _,solution = solver.solve(src=src)
elif start.mode == 'D':
    solver = DFS(map)
    _,solution = solver.solve(src=src)

visualize.visualize(solution,start,map)