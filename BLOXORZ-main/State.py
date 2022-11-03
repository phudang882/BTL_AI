from Block import Block
from Point import Point

class State:
    def __init__(self, matrix, init_status, init_fst_point, init_snd_point):
        self.matrix = matrix
        self.init_status = init_status
        self.init_fst_point = init_fst_point
        self.init_snd_point = init_snd_point


stateDemo = State(
    [
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 2, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    ],
    Block.STANDING,
    Point(1, 2),
    Point(1, 2)
)

state1 = State(
    [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    ],
    Block.STANDING,
    Point(1,1),
    Point(1,1)
)
