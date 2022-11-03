class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, pnt):
        dis = abs(pnt.x - self.x) + abs(pnt.y - self.y)
        return dis
    
    def __str__(self):
        return str(self.x) + str(self.y)