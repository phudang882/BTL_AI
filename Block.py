
import pygame
from Point import Point
class Block:
    LIE_HORIZON = 1
    LIE_VERTICAL = 2
    STANDING = 3
    SCALE_SIZE = 64
    img_standing = pygame.image.load("imgs/block_standing.png")
    img_lying = pygame.image.load("imgs/block_lying.png")

    def __init__(self,status, point_1:Point, point_2:Point):
        self.status = status
        self.point_1 = point_1
        self.point_2 = point_2

    def visualize(self, screen):
        if(self.status == Block.STANDING):
            screen.blit(self.img_standing, (self.point_1.x * Block.SCALE_SIZE, self.point_1.y * Block.SCALE_SIZE))
        else:
            screen.blit(self.img_lying, (self.point_1.x * Block.SCALE_SIZE, self.point_1.y * Block.SCALE_SIZE))
            screen.blit(self.img_lying, (self.point_2.x * Block.SCALE_SIZE, self.point_2.y * Block.SCALE_SIZE))

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