from Block import Block
from Map import Map

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