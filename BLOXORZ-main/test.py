import time
# import pygame
from Block import Block
from Map import Map
import constants as C
import copy
from State import *
from dfs import DepthFirstSearch
from GeneticAlgorithm import GeneticAlgorithm

# Set up MODE 
MODE = C.GA_MODE

# init
# pygame.init()

# # set title
# pygame.display.set_caption("BLOXORZ")
# icon = pygame.image.load("imgs/icon.png")
# pygame.display.set_icon(icon)

# state of the game
state = state1

# setting map
# cell_map = pygame.image.load("imgs/map.png")
# hole_map = pygame.image.load("imgs/hole.png")
matrix = state.matrix
map = Map(None, None , matrix)

# setting block
init_status = state.init_status
init_fst_point = state.init_fst_point
init_snd_point = state.init_snd_point
block = Block(init_status, copy.copy(init_fst_point), copy.copy(init_snd_point))

# create the screen
# screen = pygame.display.set_mode((map.width * C.CELL_SIZE, map.height * C.CELL_SIZE))

# DFS set up
# dfs = DepthFirstSearch(map)
# success, solution = dfs.solve(copy.deepcopy(block))

# GA set up
import matplotlib.pyplot as plt
generations = []
for i in range(1000):
    ga = GeneticAlgorithm(map)
    generation = ga.solve(copy.deepcopy(block))
    generations.append((i,generation))
plt.plot(*zip(*generations))
plt.show()
# start game
i = 1
running = True
# while running:
    
#     screen.fill((255, 255, 255))
#     map.draw(screen)
#     block.draw(screen)
    
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#         if MODE == C.MANUAL_MODE:
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_LEFT:
#                     block.move_left()
#                 if event.key == pygame.K_RIGHT:
#                     block.move_right()
#                 if event.key == pygame.K_UP:
#                     block.move_up()
#                 if event.key == pygame.K_DOWN:
#                     block.move_down()

#     if MODE == C.MANUAL_MODE:
#         if map.impact(block) == True:
#             block.status = init_status
#             block.fst_point = copy.copy(init_fst_point)
#             block.snd_point = copy.copy(init_snd_point)

#     if MODE == C.DFS_MODE:
#         if i < len(solution):
#             block = solution[i]
#             i += 1
#         time.sleep(0.5)

#     if MODE == C.GA_MODE:
#         if i < len(solution2):
#             block = solution2[i]
#             i += 1
#         time.sleep(0.5)


    # pygame.display.update()