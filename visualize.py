import pygame
from Block import *
import time
import copy
from Start import *
from Map import Map
def visualize(solution,start: Start,map: Map):
    screen = pygame.display.set_mode((map.width * Block.SCALE_SIZE, map.height * Block.SCALE_SIZE))
    block = Block(start.status,start.init_point_1,start.init_point_2)
    print(start.mode)
    i = 1
    running = True
    while running:
        screen.fill((255, 255, 255))
        map.visualize(screen)
        block.visualize(screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if start.mode[0] != 'G' and start.mode[0] != 'D':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        block.move_left()
                    if event.key == pygame.K_RIGHT:
                        block.move_right()
                    if event.key == pygame.K_UP:
                        block.move_up()
                    if event.key == pygame.K_DOWN:
                        block.move_down()   
            
        if start.mode[0] != 'G' and start.mode[0] != 'D':
            if map.impact(block) == True:
                block.status = start.status
                block.point_1 = copy.copy(start.init_point_1)
                block.point_2 = copy.copy(start.init_point_2)
        else:
            if i < len(solution):
                block = solution[i]
                i += 1
                time.sleep(0.5)

        pygame.display.update()