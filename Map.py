from Point import Point
from Block import Block
class Map:
    def __init__(self, background, hole, matrix):
        self.background = background
        self.hole = hole
        self.matrix = matrix
        self.height = len(matrix)
        width = self.width = len(matrix[0])
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
                    screen.blit(self.background, (pnt.x * Block.SCALE_SIZE, pnt.y * Block.SCALE_SIZE))
                elif j == 2:
                    screen.blit(self.hole, (pnt.x * Block.SCALE_SIZE, pnt.y * Block.SCALE_SIZE))
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