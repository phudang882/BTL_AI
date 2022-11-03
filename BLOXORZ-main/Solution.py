from Block import Block
from Map import Map

class Solution:

    def __init__(self, map: Map) -> None:
        self.solution = []
        self.map = map
        self.moves = ["move_left", "move_right", "move_up", "move_down"]

    def solve(self, src: Block):
        pass

    def is_goal(self, block):
        return self.map.won_the_game(block)

    def is_failed(self, block):
        return self.map.block_out_map(block) or self.map.block_felt(block)

    @staticmethod
    def print_stack(st):
        for b in st:
            print(b, end=' ')
        print()