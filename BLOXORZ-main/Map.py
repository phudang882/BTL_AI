
from Block import Block
from Point import Point
import constants as C

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
                    screen.blit(self.cell, (pnt.x * C.CELL_SIZE, pnt.y * C.CELL_SIZE))
                elif j == 2:
                    screen.blit(self.hole, (pnt.x * C.CELL_SIZE, pnt.y * C.CELL_SIZE))
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

    def won_the_game(self, block):
        won_the_game = self.matrix[block.fst_point.y][block.fst_point.x] == 2 and block.status == Block.STANDING
        if won_the_game:
            print("You Won <--(^_^)-->")
            return True
        return False

    def impact(self, block):
        return self.block_out_map(block) or self.block_felt(block) or self.won_the_game(block)

#    def hole_point(self):
        