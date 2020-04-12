import math
from array import array
import reprlib
from functools import reduce

class Vector:
    # N 维 Vector
    typecode = 'd'
    
    def __init__(self, components):
        self._components = array(self.typecode, components)
        
    # 对str提供支持    
    def __str__(self):
        return str(tuple(self))
    
    def __iter__(self):
        return (i for i in self._components)
    
    # 对 repr提供支持
    def __repr__(self):
        """
        return Vector([1.0, 2.0, 3.0...])
        """
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return f"{type(self).__name__}({components})"
    
    # 对 hash 提供支持
    def __hash__(self):
        hash_list = map(lambda x: hash(x), self._components)
        return reduce(lambda a, b: a^b, hash_list, 0)
    
    def __eq__(self, v):
        if len(self) != len(v):
            return False
        for a, b in zip(self, v):
            if a != b:
                return False
#         return tuple(self) == tuple(v)
#         return len(self) == len(self) and all(a == b for a, b in zip(self, v))
    
    # 对 abs 提供支持
    def __abs__(self):
        return math.sqrt(sum(x * x for x in self._components))
    
    # 对bool提供支持
    def __bool__(self):
        return bool(abs(self))
    
    # Python的序列协议只需要实现`__len__` ， `__getitem__`两个方法
    # 这个类是不是序列类的子类无关紧要，只需要实现这两个方法即可在用在任何期待序列的地方。

    def __len__(self):
        return len(self._components)
    
    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, int):
            return self._components[index]
        else:
            raise TypeError(f"{cls.__name__} indices must be integers.")
    
    # 运算符重载
    # +
    def __add__(self, v):
        cls = type(self)
        return cls([x + y for x, y in itertools.zip_longest(self, v, fillvalue=0)])
    
    def __radd__(self, v):
        return self + v
    
    # * scalar
    def __mul__(self, scalar):
        cls = type(self)
        return cls([x * scalar for x in self])
    
    def __rmul__(self, scalar):
        return self * scalar
        
    # dot
    def __matmul__(self, v):
        cls = type(self)
        return sum([a * b for a, b in itertools.zip_longest(self, v, fillvalue=1)])
        
    def __rmatmul__(self, v):
        return self @ v
    
    @staticmethod
    def log():
        print('ok')

class CalculabilityMixin:
    # +
    def plus(self, v):
        cls = type(self)
        return cls([x + y for x, y in itertools.zip_longest(self, v, fillvalue=0)])
    # - 
    def minus(self, v):
        cls = type(self)
        return cls([x - y for x, y in itertools.zip_longest(self, v, fillvalue=0)])
    
    # @
    def dot(self, v):
        cls = type(self)
        return sum([a * b for a, b in itertools.zip_longest(self, v, fillvalue=1)])
    

class LogMixin:
    
    def __getitem__(self, index):
        print(f"Getting value of index {index}.")
        return super().__getitem__(index)
            

class Vector2d(LogMixin, CalculabilityMixin, Vector):
    
    def __init__(self, *args):
        components = list(args) if len(args) > 1 else args[0]
        super().__init__(components)
        self._x = components[0]
        self._y = components[1]
