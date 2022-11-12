from Block import *
class Start:
    def __init__(self, matrix, init_point_1, init_point_2, mode):
        self.matrix = matrix
        self.init_point_1 = init_point_1
        self.init_point_2 = init_point_2
        self.status = Block.LIE_HORIZON
        print(init_point_1 == init_point_2)
        if init_point_1 == init_point_2:
            self.status = Block.STANDING
        elif init_point_1.x == init_point_2.x: # cùng hàng
            self.status = Block.LIE_VERTICAL
        self.mode = mode 
