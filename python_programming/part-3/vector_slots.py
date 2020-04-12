import math
from memory_profiler import profile

class Vector:
    
    __slots__ = ('__x', '__y')
    
    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)
        
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
        
    # 对str提供支持    
    def __str__(self):
        return str(tuple(self))
    
    def __iter__(self):
        return (i for i in (self.x , self.y))
    
    # 对 repr提供支持
    def __repr__(self):
        """
        return Vector(x, y)
        """
        return f"{type(self).__name__}({self.x}, {self.y})"
    
    # 对 hash 提供支持
    def __hash__(self):
        
        return hash(self.x) ^ hash(self.y)
    
    # 对 abs 提供支持
    def __abs__(self):
        return math.hypot(self.x, self.y)
    
    # 对bool提供支持
    def __bool__(self):
        return bool(abs(self))

@profile
def main():
    vectors = [Vector(1, 2) for _ in range(100000)]

main()
