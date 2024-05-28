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
eye(N): Creates an identity matrix of dimensions (N x N).
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
to_numpy(): Converts the Matrix object to a NumPy array.

"""


from typing import List,Tuple
import numpy as np

version = "0.3.0"

def matrix(array_2d: List[List[int|float]]): return Matrix(array_2d)

def zeros(dim: Tuple):
    if len(dim) == 2:
        return Matrix([[0 for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number rows and columns")

def ones(dim: Tuple):
    if len(dim) == 2:
        return Matrix([[1 for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def fill(value: int|float,dim: Tuple):
    if len(dim) == 2:
        return Matrix([[value for _ in range(dim[1])] for _ in range(dim[0])])
    raise ValueError("dimension consist only number of rows and columns")

def eye(N: int):
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

def random(dim: Tuple[int,int],seed:None|int = None):
    new_mat = []
    if seed is not None:
        np.random.seed(seed)
    for _ in range(dim[0]):
        buffer = []
        for _ in range(dim[1]):
            buffer.append(np.random.randint(0,2))
        new_mat.append(buffer)
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

    def __check__(self,array_2d):
        num_elements_in_first_row = len(array_2d[0])
        for i, row in enumerate(array_2d):
            if len(row) != num_elements_in_first_row:
                raise ValueError(f"row {i+1} does not have the same number of elements as the first row")
        return True

    def __repr__(self):
        return f"<'Matrix' object at {hex(id(self))} size={self.__size} shape={self.__shape}>"

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

    def to_numpy(self): return np.array(self.__matrix)

    def __eq__(self,other):
        if isinstance(other,(int,float)):
            for row in range(self.__row):
                for x in range(self.__col):
                    if self.__matrix[row][x] == other:
                        continue
                    else:
                        return False
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError("shape of matrices doesn't match")
            for row in range(self.__row):
                for x in range(self.__col):
                    if self.__matrix[row][x] == other.__matrix[row][x]:
                        continue
                    else:
                        return False
        else:
            raise ValueError(f"`{type(other).__name__}` isn't compatible with `Matrix`")
        return True

    def __ne__(self,other):
        if isinstance(other,(int,float)):
            for row in range(self.__row):
                for x in range(self.__col):
                    if self.__matrix[row][x] == other:
                        continue
                    else:
                        return False
        elif isinstance(other,Matrix):
            if self.__shape != other.__shape:
                raise ValueError(f"shape of matrices should be equal `{self.__shape}` != `{other.__shape}`")
            for row in range(self.__row):
                for x in range(self.__col):
                    if self.__matrix[row][x] == other.__matrix[row][x]:
                        continue
                    else:
                        return False
        else:
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")
        return True

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
        raise TypeError(f"type `{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"type `{type(other).__name__}` isn't compatible with `Matrix`")

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
            for row in range(self.__matrix):
                buffer = []
                for x in range(len(self.__matrix[0])):
                    buffer.append(self.__matrix[row][x] * other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat)
        raise TypeError(f"type `{type(other).__name__}` isn't compatible with `Matrix`")

    def __matmul__(self,other):
        if not isinstance(other, Matrix):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

    def eq(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

    def ne(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")


    def lt(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

    def gt(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

    def le(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

    def ge(self,other):
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
            raise TypeError(f"`{type(other).__name__}` isn't compatible with `Matrix`")

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
        value,vec = np.linalg.eig(self.to_numpy())
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

