class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def distance_to(self, p):
        return abs(p.x - self.x) + abs(p.y - self.y)

    def __str__(self):
        return str(self.x) + " " + str(self.y)
    
    def __eq__(self, __o) -> bool:
        return self.x == __o.x and self.y==__o.y