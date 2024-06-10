"""

Matrix (LTS)
============

This is the long term support module

The Matrix class provides a way to perform various matrix operations in Python.
This includes basic arithmetic operations, trigonometric operations, logarithmic operations,
and matrix-specific operations like transpose, determinant, inverse, eigenvalues, and eigenvectors.
It also supports operations like LU decomposition, matrix reshaping, and calculating various matrix norms.

Key Functions and Methods:

Creation Functions:
matrix(array_2d): Creates a Matrix object from a 2D list.
zeros(dim): Creates a matrix of the given dimensions filled with zeros.
ones(dim): Creates a matrix of the given dimensions filled with ones.
fill(value, dim): Creates a matrix of the given dimensions filled with a specified value.
identity(N): Creates an identity matrix of dimensions (N x N).
random: Creates a matrix with random value and shape (M x N).

Matrix Properties:
row: Returns the number of rows.
col: Returns the number of columns.
shape: Returns the shape (dimensions) of the matrix.
size: Returns the total number of elements in the matrix.
T: Returns the transpose of the matrix.

Matrix Operations:
Arithmetic Operations: add, sub, mul, truediv, floordiv, pow, mod, matmul
Trigonometric Operations: sin, cos, tan, sec, cosec, cot
Logarithmic Operations: log, ln
Exponential Operation: exp
Matrix Operations: det, inverse, trace, adjoint, transpose, reshape, flatten, sum, frobenius_norm, one_norm, inf_norm, eigen, lu_decomposition

Additional Methods:
is_square(): Checks if the matrix is square.
is_invertible(): Checks if the matrix is invertible.
is_singular(): Checks if the matrix is singular.
from_numpy(array): Creates a Matrix object from a NumPy array.
numpy(): Converts the Matrix object to a NumPy array.

"""

from typing import List,Tuple,Union
from matplotlib import pyplot as plt
import networkx as netx
import seaborn as sns
import numpy as np
import json
import csv
import os

version = "0.4.2"

def matrix(array_2d: List[List[Union[int,float]]]): return Matrix(array_2d)

def zeros(dim: Tuple):
    if len(dim) == 2:
        return Matrix([[0 for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def null(dim: Tuple[Union[int,float]]):
    if len(dim) == 3:
        return Matrix([[0 for _ in range(dim(1))] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def ones(dim: Tuple):
    if len(dim) == 2:
        return Matrix([[1 for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def fill(value: Union[int,float],dim: Tuple):
    if len(dim) == 2:
        return Matrix([[value for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def identity(N: int):
    new_mat = []
    for x in range(N):
        buffer = []
        for y in range(N):
            if x == y:
                buffer.append(1)
            else:
                buffer.append(0)
        new_mat.append(buffer)
    return Matrix(new_mat)

def diagonal(value: int|float,N: int):
    new_mat = []
    for row in range(N):
        buffer = []
        for x in range(N):
            if row == x:
                buffer.append(value)
            else:
                buffer.append(0)
        new_mat.append(buffer)
    return Matrix(new_mat)

def random(dim: Tuple[int,int],seed :None|int = None):
    new_mat = []
    if seed is not None:
        np.random.seed(seed)
    for _ in range(dim[0]):
        buffer = []
        for _ in range(dim[1]):
            buffer.append(np.random.randint(0,2))
        new_mat.append(buffer)
    return Matrix(new_mat)

def read_csv(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"given file name `{filename}` doesn't exist")
    with open(filename,"r") as file:
        reader = csv.reader(file)
        data = [list(map(float,row)) for row in reader]
    return Matrix(data)

def read_json(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"given file name `{filename}` doesn't exist")
    new_mat = []
    with open(filename,"r") as file:
        data = json.load(file)
        for x in range(len(data.keys())):
            new_mat.append(data[f"{x}"])
    return Matrix(new_mat)


class Matrix:
    def __init__(self,array_2d):
        if not self.__check__(array_2d):
            raise ValueError("specify a 2D array with an equal number of elements in each row")
        self.__matrix = array_2d
        self.__row = len(array_2d)
        self.__col = len(array_2d[0])
        self.__size = self.__row * self.__col
        self.__shape = self.__row,self.__col
        self.__iter_index = 0

    def __check__(self,array_2d):
        num_elements_in_first_row = len(array_2d[0])
        for i, row in enumerate(array_2d):
            if len(row) != num_elements_in_first_row:
                raise ValueError(f"row {i+1} does not have the same number of elements as the first row")
            for element in row:
                if not isinstance(element,(int,float,bool)):
                    raise TypeError(f"element {element} in row {i+1} is not an int or float")
        return True

    def __iter__(self):
        self.__iter_index = 0
        return self

    def __next__(self):
        if self.__iter_index < self.__size:
            row = self.__iter_index // self.__col
            col = self.__iter_index % self.__col
            self.__iter_index += 1
            return self.__matrix[row][col]
        else:
            raise StopIteration

    def __repr__(self):
        return f"<'Matrix' object at {hex(id(self))} size={self.__size} shape={self.__shape}"

    def __del__(self):
        del self

    def __getitem__(self,indices):
        if isinstance(indices,tuple):
            row,col = indices
            return self.__matrix[row][col]
        else:
            row = indices
            return self.__matrix[row]

    def __setitem__(self,indices,value):
        if isinstance(indices,tuple):
            row,col = indices
            if isinstance(value,(int,float)):
                self.__matrix[row][col] = value
            else:
                raise TypeError("expected value to be an int or float")
        else:
            row = indices
            if isinstance(value,list):
                if len(value) == self.__col:
                    self.__matrix[row] = value
                else:
                    raise ValueError("number of elements in the row must be equal to the number of columns")
            else:
                raise TypeError("expected value to be a list")

    @property
    def row(self): return self.__row

    @property
    def col(self): return self.__col
    
    @property
    def shape(self): return self.__shape
    
    @property
    def size(self): return self.__size

    @property
    def T(self):
        t = [[self.__matrix[j][i] for j in range(self.__row)] for i in range(self.__col)]
        return Matrix(t)

    @classmethod
    def from_numpy(cls,array):
        if isinstance(array,np.ndarray):
            return cls(array.tolist())
        raise TypeError("input should be a numpy array")

    def numpy(self): return np.array(self.__matrix)

    def __add__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in self.__matrix:
                buffer = []
                for x in row:
                    buffer.append(x + other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            new_mat = []
            if self.shape != other.shape:
                raise ValueError(f"shape of matrices should be equal `{self.__shape}` != `{other.__shape}`")
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] + other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for +: `{type(other).__name__}` and `Matrix`")

    def __radd__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other + self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for +: `{type(other).__name__}` and `Matrix`")

    def __sub__(self,other):
            if isinstance(other,(int,float)):
                new_mat = []
                for row in self.__matrix:
                    buffer = []
                    for x in row:
                        buffer.append(x - other)
                    new_mat.append(buffer)
                return Matrix(new_mat)
            elif isinstance(other,Matrix):
                new_mat = []
                if self.shape != other.shape:
                    raise ValueError(f"shape of matrices should be equal `{self.__shape}` != `{other.__shape}`")
                for row in range(self.__row):
                    buffer = []
                    for x in range(self.__col):
                        buffer.append(self.__matrix[row][x] - other.__matrix[row][x])
                    new_mat.append(buffer)
                return Matrix(new_mat)
            raise TypeError(f"unsupported operand type for -: `{type(other).__name__}` and `Matrix`")

    def __rsub__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other - self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupoorted operand type for -: `{type(other).__name__}` and `Matrix`")

    def __mul__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in self.__matrix:
                buffer = []
                for x in row:
                    buffer.append(x * other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            new_mat = []
            if self.__shape != other.__shape:
                raise ValueError(f"shape of matrices should be equal `{self.__shape}` != `{other.__shape}`")
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] * other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for *: `{type(other).__name__}` and `Matrix`")

    def __rmul__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other * self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for *: `{type(other).__name__}` and `Matrix`")

    def __matmul__(self,other):
        if not isinstance(other,Matrix):
            raise TypeError(f"multiplication is only supported between two matrices, not between a matrix and {type(other).__name__}")
        if self.__col != other.__row:
            raise ValueError(f"number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication")
        result_matrix = [[0 for _ in range(other.__col)] for _ in range(self.__row)]
        for i in range(self.__row):
            for j in range(other.__col):
                for k in range(self.__col):
                    result_matrix[i][j] += self.__matrix[i][k] * other.__matrix[k][j]
        return Matrix(result_matrix)

    def __truediv__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] / other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError(f"shape of the matrices must be the same")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] / other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for /: `{type(other).__name__}` and `Matrix`")

    def __rtruediv__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other / self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for /: `{type(other).__name__}` and `Matrix`")

    def __floordiv__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] // other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError(f"shape of the matrices must be the same")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] // other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for //: `{type(other).__name__}` and `Matrix`")

    def __rfloordiv__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other // self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for //: `{type(other).__name__}` and `Matrix`")

    def __pow__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] ** other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(buffer)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for **: `{type(other).__name__}` and `Matrix`")

    def __rpow__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other ** self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for **: `{type(other).__name__}` and `Matrix`")

    def __mod__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] % other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape == other.__shape:
                new_mat = []
                for row in range(self.__row):
                    buffer = []
                    for x in range(self.__col):
                        buffer.append(self.__matrix[row][x] % other.__matrix[row][x])
                    new_mat.append(buffer)
                return Matrix(new_mat)
            else:
                raise ValueError("shape of the matrices must be equal")
        else:
            raise TypeError(f"unsupported operand type for %: `{type(other).__name__}` and `Matrix`")

    def __rmod__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other % self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for %: `{type(other).__name__}` and `Matrix`")

    def __pos__(self): return Matrix(self.__matrix)

    def __neg__(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(-self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat)

    def __and__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] & other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("shape of the matrices must be equal")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] & other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for &: `{type(other).__name__}` and `Matrix`")
        
    def __rand__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other & self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for &: `{type(other).__name__}` and `Matrix`")

    def __or__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] | other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("shape of the matrices must be equal")
        else:
            raise TypeError(f"unsupported operand type for |: {type(other).__name__}` and `Matrix`")

    def __ror__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other | self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for |: `{type(other).__name__}` and `Matrix`")

    def __invert__(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(~self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat)

    def __xor__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] ^ other)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("shape of the matrices must be equal")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(self.__matrix[row][x] ^ other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for ^: `{type(other).__name__}` and `Matrix`")

    def __rxor__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(other ^ self.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"unsupported operand type for ^: `{type(other).__name__}` and `Matrix`")

    def __eq__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] == other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] == other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for ==: `{type(other).__name__}` and `Matrix`")

    def __ne__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] != other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] != other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for !=: {type(other).__name__}` and `Matrix`")
        
    def __lt__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] < other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] < other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for <: `{type(other).__name__}` and `Matrix`")
        
    def __gt__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] > other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] > other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for >:`{type(other).__name__}` and `Matrix`")

    def __le__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] <= other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] <= other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for <=: `{type(other).__name__}` and `Matrix`")

    def __ge__(self,other):
        if isinstance(other,(int,float)):
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] >= other:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("can't compare two matrices with different shapes")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    if self.__matrix[row][x] >= other.__matrix[row][x]:
                        buffer.append(True)
                    else:
                        buffer.append(False)
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise TypeError(f"unsupported operand type for >=: `{type(other).__name__}` and `Matrix`")

    def reciprocate(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(1 / self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat)

    def is_null(self):
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] == 0:
                    continue
                else:
                    return False
        return True

    def mat_pow(self,n: int):
        if not isinstance(n,int):
            raise TypeError(f"given base must be an integer not `{type(n).__name__}`")
        elif n < 0:
            raise ValueError("`n` must be greater than 0")
        result = Matrix([[1 if i == j else 0 for j in range(self.__col)] for i in range(self.__row)])
        base = self
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
        return result

    def pos(self): return +self

    def neg(self): return -self

    def add(self,other): return self + other

    def sub(self,other): return self - other

    def mul(self,other): return self * other

    def matmul(self,other): return self @ other

    def mod(self,other): return self % other

    def pow(self,other): return self ** other

    def truediv(self,other): return self / other

    def floordiv(self,other): return self // other

    def AND(self,other): return self & other

    def NAND(self,other): return ~(self & other)

    def OR(self,other): return self | other

    def NOR(self,other): return ~(self | other)

    def INVERT(self): return ~self

    def XOR(self,other): return self ^ other

    def XNOR(self,other): return ~(self ^ other)

    def eq(self,other): return  self == other

    def ne(self,other): return self != other

    def gt(self,other): return self > other

    def lt(self,other): return self < other

    def ge(self,other): return self >= other

    def le(self,other): return self <= other

    def type_cast(self,dtype: type,inplace: bool = False):
        if inplace:
            for row in range(self.__row):
                for x in range(self.__col):
                    self.__matrix[row][x] = dtype(self.__matrix[row][x])
        elif not inplace:
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col):
                    buffer.append(dtype(self.__matrix[row][x]))
                new_mat.append(buffer)
            return Matrix(new_mat)
        else:
            raise ValueError("invalid argument for `dtype` expected from `True` or `False`")

    def copy(self):
        return Matrix([row[:] for row in self.__matrix])

    def get(self,row: int,col: None|int = None):
        if col is None:
            return self.__matrix[row]
        else:
            return self.__matrix[row][col]

    def set(self,row: int,value: List[int | float] | int | float,col: None|int = None):
        if col is None:
            if isinstance(value,list) and len(value) == self.__col:
                self.__matrix[row] = value
                return
            raise TypeError(f"can't set `{type(value).__name__}` to matrix row")
        else:
            self.__matrix[row][col] = value

    def concate(self,other,axis = 0):
        if isinstance(other, Matrix):
            if axis == 0:
                if self.__col != other.__col:
                    raise ValueError("Matrices must have the same number of columns to concatenate vertically")
                new_matrix = self.__matrix + other.__matrix
            elif axis == 1:
                if self.__row != other.__row:
                    raise ValueError("Matrices must have the same number of rows to concatenate horizontally")
                new_matrix = [self.__matrix[row] + other.__matrix[row] for row in range(self.__row)]
            else:
                raise ValueError("Axis must be 0 (vertical) or 1 (horizontal)")
        elif isinstance(other, list):
            if axis == 0:
                if len(other) != self.__col:
                    raise ValueError("The list must have the same number of elements as there are columns in the matrix to concatenate vertically")
                new_matrix = self.__matrix + [other]
            elif axis == 1:
                if len(other) != self.__row:
                    raise ValueError("the list must have the same number of elements as there are rows in the matrix to concatenate horizontally")
                new_matrix = [self.__matrix[row] + [other[row]] for row in range(self.__row)]
            else:
                raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")
        else:
            raise TypeError(f"couldn't add `{type(other).__name__}` and `Matrix`, other must be `list` or `Matrix`")
        return Matrix(new_matrix)

    def to_json(self,file_path: str):
        if os.path.exists(file_path):
            raise FileExistsError(f"file with name `{file_path}` already exist!")
        with open(file_path,"w") as file:
            file.write("{\n")
            for row in range(self.__row):
                if row == self.__row - 1:
                    file.write(f'   "{row}": {self.__matrix[row]}\n')
                else:
                    file.write(f'   "{row}": {self.__matrix[row]},\n')
            file.write("}\n")
            file.close()

    def to_csv(self,file_path: str):
        if os.path.exists(file_path):
            raise FileExistsError(f"file with name `{file_path}` already exist!")
        with open(file_path,"w") as file:
            for row in range(self.__row):
                for x in range(self.__col):
                    if x == self.__col - 1:
                        file.write(f"{self.__matrix[row][x]}")
                    else:
                        file.write(f"{self.__matrix[row][x]},")
                file.write("\n")
            file.close()

    def to_dict(self):
        result = {}
        for row in range(self.__row):
            result[f"{row}"] = self.__matrix[row]
        return result

    def to_list(self):
        return self.__matrix

    def mean(self,axis = 1):
        if axis == 1:
            new_mat = []
            for row in range(self.__row):
                total = 0
                for x in range(self.__col):
                    total += self.__matrix[row][x]
                new_mat.append([total / self.__col])
            return Matrix(new_mat)
        elif axis == 0:
            new_mat = []
            for x in range(self.__col):
                total = 0
                for row in range(self.__row):
                    total += self.__matrix[row][x]
                new_mat.append([total / self.__row])
            return Matrix(new_mat)
        else:
            raise ValueError("invalid axis given!")

    def var(self,axis = 1):
        if axis == 1:
            new_mat = []
            for row in range(self.__row):
                total = 0
                for x in range(self.__col):
                    total += (self.__matrix[row][x] - self.mean(axis = 1)) ** 2.0
                new_mat.append([total / self.__col])
            return Matrix(new_mat)
        elif axis == 0:
            new_mat = []
            for x in range(self.__col):
                total = 0
                for row in range(self.__row):
                    total += (self.__matrix[row][x] - self.mean(axis = 0)) ** 2.0
                new_mat.append([total / self.__row])
            return Matrix(new_mat)
        else:
            raise ValueError("invalid axis given!")

    def std(self,axis = 1): return Matrix(self.var(axis = axis).sqrt())

    def median(self,axis = 1):
        if axis == 1:
            new_mat = []
            for row in range(self.__row):
                sorted_row = sorted(self.__matrix[row])
                mid = self.__col // 2
                if self.__col % 2 == 0:
                    median_value = (sorted_row[mid - 1] + sorted_row[mid]) / 2.0
                else:
                    median_value = sorted_row[mid]
                new_mat.append([median_value])
            return Matrix(new_mat)
        elif axis == 0:
            new_mat = []
            for x in range(self.__col):
                col_values = [self.__matrix[row][x] for row in range(self.__row)]
                sorted_col = sorted(col_values)
                mid = self.__row // 2
                if self.__row % 2 == 0:
                    median_value = (sorted_col[mid - 1] + sorted_col[mid]) / 2.0
                else:
                    median_value = sorted_col[mid]
                new_mat.append(median_value)
            return Matrix([new_mat])
        else:
            raise ValueError("invalid axis given!")

    def max(self,axis: None|int = None):
        if axis is None:
            result = -float("inf")
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result < value:
                        result = value
            return result
        elif axis == 0:
            result = [-float("inf")] * self.__col
            for x in range(self.__col):
                for row in range(self.__row):
                    value = self.__matrix[row][x]
                    if result[x] < value:
                        result[x] = value
            return result
        elif axis == 1:
            result = [-float("inf")] * self.__row
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result[row] < value:
                        result[row] = value
            return result
        else:
            raise ValueError("invalid argument for axis it can either be None, 0 or 1")
    
    def min(self,axis: None|int = None):
        if axis is None:
            result = float("inf")
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result > value:
                        result = value
            return result
        elif axis == 0:
            result = [float("inf")] * self.__col
            for x in range(self.__col):
                for row in range(self.__row):
                    value = self.__matrix[row][x]
                    if result[x] > value:
                        result[x] = value
            return result
        elif axis == 1:
            result = [float("inf")] * self.__row
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result[row] > value:
                        result[row] = value
            return result
        else:
            raise ValueError("invalid argument for axis it can either be None, 0 or 1")
    
    def sort(self,axis = 1):
        if axis == 1:
            new_mat = [sorted(row) for row in self.__matrix]
            return Matrix(new_mat)
        elif axis == 0:
            new_mat = [[self.__matrix[row][col] for row in range(self.__row)] for col in range(self.__col)]
            sorted_mat = [sorted(col) for col in new_mat]
            transposed_sorted_mat = [[sorted_mat[col][row] for col in range(self.__col)] for row in range(self.__row)]
            return Matrix(transposed_sorted_mat)
        else:
            raise ValueError("invalid axis given!")

    def sqrt(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(self.__matrix[row][x] ** 0.5)
            new_mat.append(buffer)
        return Matrix(new_mat)

    def cbrt(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(self.__matrix[row][x] ** (1/3))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def floor(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                number = self[row][x]
                if number >= 0:
                    buffer.append(int(number))
                else:
                    if int(number) == number:
                        buffer.append(number)
                    else:
                        buffer.append(int(number) - 1)
            new_mat.append(buffer)
        return Matrix(new_mat)

    def ceil(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                number = self[row][x]
                if number == int(number):
                    buffer.append(int(number))
                else:
                    buffer.append(int(number) + 1 if number > 0 else int(number))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def scale(self,scalar):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(self.__matrix[row][x] * scalar)
            new_mat.append(buffer)
        return Matrix(new_mat)

    def log(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(np.log10(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def ln(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(np.log(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def sin(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(np.sin(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def cos(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(np.cos(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def tan(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(np.tan(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def sec(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(1 / np.cos(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def cosec(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(1 / np.sin(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def cot(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(1 / np.tan(self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat)

    def exp(self):
        E = 2.7182818284590452353602874713527
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(E ** self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat)

    def eigen(self):
        if not self.is_square():
            raise ValueError("eigen values and its vectors are defined only for square matrices")
        value,vec = np.linalg.eig(self.numpy())
        return value,Matrix(vec.tolist())

    def cofactor(self,i,j):
        sub_matrix = [row[:j] + row[j + 1:] for row in (self.__matrix[:i] + self.__matrix[i + 1:])]
        sign = (-1) ** (i + j)
        return sign * Matrix(sub_matrix).det()

    def row_matrix(self): return self.__col == 1

    def column_matrix(self): return self.__row == 1
    
    def minor(self,i,j):
        sub_matrix = [row[:j] + row[j+1:] for row in (self.__matrix[:i] + self.__matrix[i+1:])]
        return Matrix(sub_matrix).det()

    def det(self):
        if not self.is_square():
            raise ValueError("determinant is defined only for square matrices")
        if self.__row == 1:
            return self.__matrix[0][0]
        if self.__row == 2:
            return self.__matrix[0][0] * self.__matrix[1][1] - self.__matrix[0][1] * self.__matrix[1][0]
        det = 0
        for j in range(self.__col):
            det += self.__matrix[0][j] * self.cofactor(0,j)
        return det

    def inverse(self):
        det = self.det()
        if det == 0:
            raise ZeroDivisionError(f"inverse of a matrix with 'zero' determinant doesn't exist!")
        return self.adjoint().scale(1/det)
    
    def is_invertible(self):
        if not self.is_square():
            raise ValueError("matrix must have equal number of rows and columns")
        return self.det() != 0

    def is_symmetric(self):
        if self.__row != self.__col:
            return False
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] != self.__matrix[x][row]:
                    return False
        return True

    def is_skew_symmetric(self):
        if self.__row != self.__col:
            return False
        for i in range(self.__row):
            for j in range(self.__col):
                if self.__matrix[i][j] != -self.__matrix[j][i]:
                    return False
        return True

    def is_singular(self):
        if not self.is_square():
            raise ValueError("matrix must have equal number of rows and columns")
        return self.det() == 0

    def adjoint(self):
        if not self.is_square():
            raise ValueError("adjoint is defined only for square matrices")
        cofactors = [[self.cofactor(i, j) for j in range(self.__col)] for i in range(self.__row)]
        return Matrix(cofactors).T

    def is_square(self): return self.__row == self.__col

    def trace(self):
        if not self.is_square():
            raise ValueError("given matrix must be a square matrix number of rows and columns must be the same")
        total = 0
        for x in range(self.__row):
            for y in range(self.__col):
                if x == y: total += self.__matrix[x][y]
        return total

    def transpose(self): return self.T

    def flatten(self):
        flattened = [item for row in self.__matrix for item in row]
        return Matrix([flattened])

    def reshape(self,dim):
        total_elements = self.__row * self.__col
        if total_elements != dim[0] * dim[1]:
            raise ValueError(f"cannot reshape array of size {total_elements} into shape {dim}")
        flattened = self.flatten().__matrix[0]
        reshaped = []
        for i in range(dim[0]):
            row = []
            for j in range(dim[1]):
                row.append(flattened[i * dim[1] + j])
            reshaped.append(row)
        return Matrix(reshaped)

    def lu_decomposition(self):
        if not self.is_square():
            raise ValueError("LU decomposition is only defined for square matrices")
        n = self.__row
        L = zeros((n,n))
        U = zeros((n,n))
        for i in range(n):
            L[i,i] = 1
            for j in range(n):
                U[i,j] = self.__matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(n):
                L[j,i] = (self.__matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i,i]
        return L,U

    def sum(self,keepdim = False):
        if keepdim:
            new_mat = []
            for row in self.__matrix:
                new_mat.append([sum(row)])
            return Matrix(new_mat)
        elif not keepdim:
            total = 0
            for row in self.__matrix:
                for x in row:
                    total += x
            return total
        else:
            raise ValueError("invalid argument")

    def frobenius_norm(self):
        sum_of_sqrs = sum(x**2 for row in self.__matrix for x in row)
        return np.sqrt(sum_of_sqrs)

    def one_norm(self):
        col_sums = [sum(abs(x) for x in column) for column in zip(*self.__matrix)]
        return max(col_sums)

    def inf_norm(self):
        row_sums = [sum(abs(x) for x in row) for row in self.__matrix]
        return max(row_sums)

    def cholesky(self):
        if self.__row != self.__col:
            raise ValueError("Matrix must be square for Cholesky decomposition")
        dense = np.array(self.__matrix)
        if not np.allclose(dense, dense.T):
            raise ValueError("Matrix must be symmetric for Cholesky decomposition")
        L = np.zeros_like(dense)
        for i in range(self.__row):
            for j in range(i + 1):
                sum_k = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = np.sqrt(dense[i][i] - sum_k)
                else:
                    L[i][j] = (dense[i][j] - sum_k) / L[j][j]
        return Matrix(L.tolist())

    def rank(self):
        mat = self.__matrix.copy()
        rank = 0
        for i in range(min(self.__row,self.__col)):
            pivot_row = None
            for j in range(i,self.__row):
                if mat[j][i] != 0:
                    pivot_row = j
                    break
            if pivot_row is not None:
                mat[i],mat[pivot_row] = mat[pivot_row],mat[i]
                rank += 1
                for j in range(i + 1,self.__row):
                    factor = mat[j][i] / mat[i][i]
                    for k in range(i,self.__col):
                        mat[j][k] -= factor * mat[i][k]
        return rank

    def heatmap(self,title = "Matrix HeatMap",cmap = "viridis"):
        plt.figure(figsize = (8,6))
        sns.heatmap(self.__matrix,annot = True,fmt = "d",cmap = cmap,cbar = True)
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def graph(self,title = "Graph Visualization of Matrix"):
        G = netx.Graph()
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] != 0:
                    G.add_edge(row,x,weight = self.__matrix[row][x])
        pos = netx.spring_layout(G)
        edge_labels = netx.get_edge_attributes(G,"weight")
        plt.figure(figsize = (8,6))
        netx.draw(G,pos,with_labels = True,node_color = "orange",edge_color = "blue",node_size = 500,alpha = 0.8)
        netx.draw_networkx_edge_labels(G,pos,edge_labels = edge_labels,font_color = "red")
        plt.title(title)
        plt.show()

