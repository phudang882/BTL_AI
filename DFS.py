from Movement import *
import copy
class DFS(Movement):

    def __init__(self, map: Map) -> None:
        super().__init__(map)
        self.stack = []
        self.path = dict()
        self.visited = set()

    def solve(self, src : Block) :
        self.stack.append(src)
        self.visited.add(src.encode())
        while len(self.stack) > 0:
            u = self.stack.pop()
            if self.move_to_goal(u):
                solution = []
                while u is not None:
                    solution += [u]
                    if u.encode() in self.path:
                        u = self.path[u.encode()]
                    else:
                        break
                    
                solution.reverse()
                self.print_stack(solution)
                return True, list(solution)

            prev_state = copy.deepcopy(u)
            
            for move_name in self.moves:
                print(move_name, end = "---\n")
                move = getattr(Block, move_name)
                move(u)
                if not self.move_out(u):
                    encode = u.encode()
                    if encode not in self.visited:
                        self.stack.append(u)
                        self.path[encode] = prev_state
                        self.visited.add(encode)
                u = copy.deepcopy(prev_state)

        return False, []