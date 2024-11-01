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
from1d(array_1d): Create a Matrix object from a 1D array with specified shape.
from2d(array_2d): Creates a Matrix object from a 2D array.
zeros(shape): Creates a matrix of the given dimensions filled with zeros.
ones(shape): Creates a matrix of the given dimensions filled with ones.
null(shape): Creates a matrix of the given dimensions filled with null(nan) values
fill(value, shape): Creates a matrix of the given dimensions filled with a specified value.
identity(N): Creates an identity matrix of dimensions (N x N).
random: Creates a matrix with random value and shape (M x N).
zeros_like(mat): Creates a matrix from the given matrix with zeros
ones_like(mat): Creates a matrix from the given matrix with ones
rand_like(mat): Creates a matrix from the given matrix with random numbers
null_like(mat): Creates a matrix from the given matrix with null(nan) values
fill_like(mat): Creates a matrix form the given matrix with a value as an argument

Matrix Properties:
symbol: Returns the unique symbol of that specific matrix
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


from __future__ import annotations
from typing import List,Tuple,Union,Optional,Type
from matplotlib import pyplot as plt
import networkx as netx
import seaborn as sns
import numpy as np
import warnings
import json
import csv
import os

version = "0.8.2"
__mem__ = {}

def from2d(
        array_2d:List[List[Union[int,float,bool,complex]]],
        dtype:Type,
        symbol:Optional[str]=None,
        overwrite:bool=False
    ): return Matrix(array_2d, dtype=dtype, symbol=symbol, overwrite=overwrite)

def from1d(
        array_1d:List[Union[int,float,bool,complex]],
        shape:Tuple[int,int],
        dtype:Type,
        symbol:Optional[str]=None,
        overwrite:bool=False
    ):
    if len(array_1d) != shape[0] * shape[1]: raise ValueError(f"Can't create a matrix with shape {shape}, {len(array_1d)} != {shape[0] * shape[1]}")
    matrix_2d = []
    for i in range(shape[0]):
        row = array_1d[i * shape[1]:(i + 1) * shape[1]]
        matrix_2d.append(row)
    return Matrix(matrix_2d, dtype=dtype, symbol=symbol, overwrite=overwrite)

def concat(
        matrices:List[Matrix],
        axis:int=0,
        symbol:Optional[str]=None
    ):
    if axis not in [None,0,1]: raise ValueError("axis must be 0 (row-wise) or 1 (column-wise)")
    if not matrices: raise ValueError("at least one matrix or list is required for concatenation.")
    first_matrix = matrices[0]
    if isinstance(first_matrix,Matrix):
        new_matrix = first_matrix.to_list()
        reference_row, reference_col = first_matrix.shape
    elif isinstance(first_matrix,list):
        new_matrix = first_matrix
        reference_row, reference_col = len(first_matrix), len(first_matrix[0]) if first_matrix else 0
    else:  raise TypeError(f"unsupported type {type(first_matrix).__name__} for concatenation.")
    for other in matrices[1:]:
        if isinstance(other,Matrix):
            other_matrix = other.to_list()
            other_row, other_col = other.shape
        elif isinstance(other,list):
            other_matrix = other
            other_row, other_col = len(other),len(other[0]) if other else 0
        else: raise TypeError(f"unsupported type {type(other).__name__} for concatenation.")
        if axis == 0:
            if reference_col != other_col: raise ValueError("all matrices or lists must have the same number of columns to concatenate vertically.")
            new_matrix += other_matrix
        elif axis == 1:
            if reference_row != other_row: raise ValueError("all matrices or lists must have the same number of rows to concatenate horizontally.")
            new_matrix = [new_matrix[row] + other_matrix[row] for row in range(reference_row)]
    return Matrix(new_matrix, symbol=symbol)

def arange(
        start:int,
        end:int,
        step:int=1,
        endpoint:bool=False,
        shape:Optional[Tuple[int,int]]=None,
        symbol:Optional[str]=None
    ):
    if endpoint: end += 1
    if shape is None:  return Matrix([[x for x in range(start,end,step)]],symbol)
    else: return Matrix([[x for x in range(start,end,step)]], symbol=symbol).reshape(shape)

def linspace(
        start:int,
        end:int,
        num:int=50,
        endpoint:bool=False,
        shape:Optional[Tuple[int,int]]=None,
        symbol:Optional[str]=None
    ):
    if num <= 0: return Matrix([[]])
    if endpoint: step = (end - start) / (num - 1)
    else: step = (end - start) / num
    linspace_values = [start + step * i for i in range(num)]
    if not endpoint: linspace_values = linspace_values[:-1]
    if shape is None: return Matrix([linspace_values],symbol)
    return Matrix([linspace_values], symbol=symbol).reshape(shape)

def __res_dtype__(matrix:Matrix, other:Matrix):
        if not isinstance(other, Matrix):
            if matrix.dtype == "complex" or isinstance(other,complex): return complex
            elif matrix.dtype == "bool" and isinstance(other,bool): return bool
            elif (matrix.dtype == "int" and isinstance(other,float)) or (matrix.dtype == "float" and isinstance(other,int)): return float
            elif (matrix.dtype == "bool" and isinstance(other,int)) or (matrix.dtype == "int" and isinstance(other,bool)): return int
            else: return float
        else:
            if matrix.dtype == "complex" or other.dtype == "complex": return complex
            elif matrix.dtype == "bool" and other.dtype == "bool": return bool
            elif (matrix.dtype == "int" and other.dtype == "float") or (matrix.dtype == "float" and other.dtype == "int"): return float
            elif (matrix.dtype == "bool" and other.dtype == "int") or (matrix.dtype == "int" and other.dtype == "bool"): return int
            else: return float

def __res_dtype_self__(matrix:Matrix):
    if matrix.dtype == "int": return int
    elif matrix.dtype == "float": return float
    elif matrix.dtype == "bool": return bool
    elif matrix.dtype == "complex": return complex

def add(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] + obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] + obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def sub(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] - obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] - obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def mul(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] * obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] * obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def matmul(
        matrix:Matrix,
        obj:Matrix,
        symbol:Optional[str]=None,   
    ):
    if matrix.col != obj.row: raise ValueError(f"number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication")
    result_matrix = [[0 for _ in range(obj.col)] for _ in range(matrix.row)]
    for i in range(matrix.row):
        for j in range(obj.col):
            for k in range(matrix.col): result_matrix[i][j] += matrix[i][k] * obj[k][j]
    return Matrix(result_matrix, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def tdiv(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] / obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] / obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def fdiv(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] // obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] // obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def mod(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] % obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] % obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def pow(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] ** obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] ** obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype__(matrix,obj), symbol=symbol)

def abs(
        matrix:Matrix,
        symbol:Optional[str]=None
    ):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            if matrix[row][col] >= 0: buffer.append(matrix[row][col])
            else: buffer.append(-matrix[row][col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def pos(
        matrix:Matrix,
        symbol:Optional[str]=None
    ):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def neg(
        matrix:Matrix,
        symbol:Optional[str]=None
    ):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(-matrix[row][col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def true_div(matrix:Matrix, obj:Union[Matrix,int,float,bool,complex], symbol:Optional[str]=None): tdiv(matrix=matrix, obj=obj, symbol=symbol)
def floor_div(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): fdiv(matrix=matrix, obj=obj, symbol=symbol)
def modulo(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): mod(matrix=matrix, obj=obj, symbol=symbol)
def power(matrix:Matrix, obj:Union[Matrix,int,float,bool,complex], symbol:Optional[str]=None): pow(matrix=matrix, obj=obj, symbol=symbol)
def absolute(matrix:Matrix, symbol:Optional[str]=None): abs(matrix=matrix, symbol=symbol)
def positive(matrix:Matrix, symbol:Optional[str]=None): pow(matrix=matrix, symbol=symbol)
def negative(matrix:Matrix, symbol:Optional[str]=None): neg(matrix=matrix, symbol=symbol)

def eq(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool,complex],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] == obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] == obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def ne(
       matrix:Matrix,
       obj:Union[Matrix,int,float,bool,complex],
       symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] != obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] != obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def gt(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool],
        symbol:Optional[str]=None    
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] > obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] > obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def ge(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] >= obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] >= obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def lt(
       matrix:Matrix,
       obj:Union[Matrix,int,float,bool],
       symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] < obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] < obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def le(
        matrix:Matrix,
        obj:Union[Matrix,int,float,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] <= obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] <= obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def equal(matrix:Matrix, obj:Union[Matrix,int,float,bool,complex], symbol:Optional[str]=None): return eq(matrix=matrix, obj=obj, symbol=symbol)
def not_equal(matrix:Matrix, obj:Union[Matrix,int,float,bool,complex], symbol:Optional[str]=None): return ne(matrix=matrix, obj=obj, symbol=symbol)
def greater(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): return gt(matrix=matrix, obj=obj, symbol=symbol)
def greater_equal(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): return ge(matrix=matrix, obj=obj, symbol=symbol)
def lesser(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): return lt(matrix=matrix, obj=obj, symbol=symbol)
def lesser_equal(matrix:Matrix, obj:Union[Matrix,int,float,bool], symbol:Optional[str]=None): return le(matrix=matrix, obj=obj, symbol=symbol)

def bitwise_and(
        matrix:Matrix,
        obj:Union[Matrix,int,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] & obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] & obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_or(
        matrix:Matrix,
        obj:Union[Matrix,int,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the twomatrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] | obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] | obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_invert(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(~matrix[row][col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_nand(
        matrix:Matrix,
        obj:Matrix,
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(~(matrix[row][col] & obj[row][col]))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(~(matrix[row][col] & obj))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_nor(
        matrix:Matrix,
        obj:Union[Matrix,int,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(~(matrix[row][col] | obj[row][col]))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(~(matrix[row][col] | obj))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_xor(
        matrix:Matrix,
        obj:Union[Matrix,int,bool],
        symbol:Optional[str]=None,
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] ^ obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] ^ obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_xnor(
        matrix:Matrix,
        obj:Union[Matrix,int,bool],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(~(matrix[row][col] ^ obj[row][col]))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(~(matrix[row][col] ^ obj))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_left_shift(
        matrix:Matrix,
        obj:Union[Matrix,int],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] << obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] << obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def bitwise_right_shift(
        matrix:Matrix,
        obj:Union[Matrix,int],
        symbol:Optional[str]=None
    ):
    if isinstance(obj,Matrix):
        if not (matrix.shape == obj.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {obj.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(matrix[row][col] >> obj[row][col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(matrix[row][col] >> obj)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int, symbol=symbol)

def empty(shape:Tuple[int,int] ,symbol:Optional[str]=None):
    row,col = shape
    new_mat = 0
    for x in range(row):
        buffer = []
        for y in range(col): buffer.append(0)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=float, symbol=symbol)

def __fill(shape:Tuple[int,int], value:Union[int,float,bool,complex], symbol:Optional[str]=None):
    # NOTE just a helper function
    if len(shape) == 2: return Matrix([[value for _ in range(shape[1])] for _ in range(shape[0])], dtype=type(value), symbol=symbol)
    raise ValueError("dimension must consist only number of rows and columns")

def ones(shape:Tuple[int,int], symbol:Optional[str]=None): return __fill(shape,1,symbol)
def zeros(shape:Tuple[int,int], symbol:Optional[str]=None): return __fill(shape,0,symbol)
def null(shape:Tuple[int,int], symbol:Optional[str]=None): return __fill(shape,0,symbol)
def fill(shape:Tuple[int,int], value:Union[int,float,bool], symbol:Optional[str]=None): return __fill(shape,value,symbol)
def ones_like(mat:Matrix, symbol:Optional[str]=None):return ones((mat.row,mat.col),symbol)
def zeros_like(mat:Matrix, symbol:Optional[str]=None): return zeros((mat.row,mat.col),symbol)
def null_like(mat:Matrix, symbol:Optional[str]=None): return null((mat.row,mat.col),symbol)
def fill_like(mat:Matrix, value:Union[int,float,bool], symbol:Optional[str]=None): return fill((mat.row,mat.col),value,symbol)
def rand_like(mat:Matrix, seed:Optional[int]=None, symbol:Optional[str]=None):
    res_dtype = complex if mat.dtype == complex else float
    return rand((mat.row,mat.col),dtype=res_dtype,seed=seed,symbol=symbol)

def isnan(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row][col]
            if value != value: buffer.append(True)
            else: buffer.append(False)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def isinf(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row][col]
            if value == float("inf") or value == -float("inf"): buffer.append(True)
            else: buffer.append(False)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def isposinf(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row][col]
            if value == float("inf"): buffer.append(True)
            else: buffer.append(False)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def isneginf(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row][col]
            if value == -float("inf"): buffer.append(True)
            else: buffer.append(False)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def isfinite(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row][col]
            if value == float("inf") or value == -float("inf") or value != value: buffer.append(False)
            else: buffer.append(True)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=bool, symbol=symbol)

def clip(
        matrix:Matrix,
        min_value:Union[int,float],
        max_value:Union[int,float],
        inplace:bool=False,
        symbol:Optional[str]=None
    ):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(min(max(matrix[row][col],min_value),max_value))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def clamp(
        matrix:Matrix,
        min_value:Union[int,float],
        max_value:Union[int,float],
        symbol:Optional[str]=None    
    ): return clip(matrix, min_value=min_value, max_value=max_value, symbol=symbol)

def identity(N:int, symbol:Optional[str]=None): 
    new_mat = []
    for x in range(N):
        buffer = []
        for y in range(N):
            if x == y: buffer.append(1)
            else: buffer.append(0)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=int,symbol=symbol)

def diagonal(
        value:int|float,
        N:int,
        symbol:Optional[str]=None
    ): 
    new_mat = []
    for row in range(N):
        buffer = []
        for x in range(N):
            if row == x: buffer.append(value)
            else: buffer.append(0)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=type(value), symbol=symbol)

def rand(
        shape:Tuple[int,int],
        dtype:Type=float,
        seed:None|int=None,
        symbol:Optional[str]=None
    ):
    new_mat = []
    if seed is not None: np.random.seed(seed)
    for _ in range(shape[0]):
        buffer = []
        for _ in range(shape[1]):
            if dtype == float: buffer.append(np.random.rand())
            elif dtype == complex:
                real_part = np.random.rand()
                imag_part = np.random.rand()
                buffer.append(complex(real_part,imag_part))
            else: raise ValueError("unsupported dtype. Use int, float, or complex.")
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=dtype, symbol=symbol)

def read_csv(filename:str, symbol:Optional[str]):
    if not os.path.exists(filename): raise FileNotFoundError(f"given file name `{filename}` doesn't exist")
    with open(filename,"r") as file:
        reader = csv.reader(file)
        data = [list(map(float,row)) for row in reader]
    return Matrix(data, symbol=symbol)

def read_json(filename:str, symbol:Optional[str]=None):
    if not os.path.exists(filename): raise FileNotFoundError(f"given file name `{filename}` doesn't exist")
    new_mat = []
    with open(filename,"r") as file:
        data = json.load(file)
        for x in range(len(data.keys())):
            new_mat.append(data[f"{x}"])
    return Matrix(new_mat, symbol=symbol)

def mem(): 
    string = ""
    keys = __mem__.keys()
    for x in keys: string += f"{x} = {__mem__[x]}\n"
    return string

def from_symbol(symbol:str): return __mem__[symbol]


class Matrix:
    def __init__(
            self,
            array_2d:List[List[Union[int,float,bool,complex]]],
            /,
            dtype:Type=float,
            symbol:Optional[str]=None,
            overwrite:bool=False
        ):
        self.__check__(array_2d,dtype,symbol,overwrite)
        self.__matrix = array_2d
        self.__dtype = dtype
        self.__symbol = symbol
        self.__row = len(array_2d)
        self.__col = len(array_2d[0])
        self.__size = self.__row * self.__col
        self.__shape = self.__row,self.__col
        self.__iter_index = 0
        if self.__symbol:
            if "__mem__" not in globals():
                global __mem__
                __mem__ = {}
            __mem__[self.__symbol] = self

    def __check__(
            self,
            array2d:List[List[Union[int,float,bool,complex]]],
            dtype:Type,
            symbol:Optional[str]=None,
            overwrite:bool=False
        ):
        no_rows = len(array2d[0])
        for index,row in enumerate(array2d):
            if len(row) != no_rows: raise ValueError(f"row {index+1} doesn't have the same number of elements as the first row")
            for elem in row:
                if isinstance(elem,str): raise TypeError(f"element '{elem}' in row {index+1} with data type '{type(elem).__name__}' isn't compatible with 'Matrix'")
        if dtype:
            for idx1,row in enumerate(array2d):
                for idx2,x in enumerate(row): array2d[idx1][idx2] = dtype(x)
        else: raise TypeError(f"Cannot convert element {x} at position {idx1+1},{idx2+1} to dtype `{dtype.__name__}`")
        if symbol is not None and not isinstance(symbol,str): raise ValueError(f"Symbol must be a string or None, not '{type(symbol).__name__}'.")
        if symbol is not None and not symbol in globals().get("__mem__",{}) and overwrite is True:
            del __mem__[symbol]
            __mem__[symbol] = self()
        if symbol is not None and symbol in globals().get("__mem__",{}) and overwrite is False: raise KeyError(f"Symbol '{symbol}' already exists! Try a different symbol.")

    def __res_dtype__(self, other):
        if not isinstance(other, Matrix):
            if self.__dtype == complex or other.__dtype == complex: return complex
            elif self.__dtype == bool and other.__dtype == bool: return bool
            elif (self.__dtype == int and other.__dtype == float) or (self.__dtype == float and other.__dtype == int): return float
            elif (self.__dtype == bool and other.__dtype == int) or (self.__dtype == int and other.__dtype == bool) or (self.__dtype == int and other.__dtype == int): return int
            else: return float
        else:
            if self.__dtype == complex or type(other) == complex: return complex
            elif self.__dtype == bool and type(other) == bool: return bool
            elif (self.__dtype == int and type(other) == float) or (self.__dtype == float and type(other) == int): return float
            elif (self.__dtype == bool and type(other) == int) or (self.__dtype == int and type(other) == bool) or (self.__dtype == int and type(other) == int): return int
            else: return float

    def __iter__(self):
        self.__iter_index = 0
        return self

    def __next__(self):
        if self.__iter_index < self.__size:
            row = self.__iter_index // self.__col
            col = self.__iter_index % self.__col
            self.__iter_index += 1
            return self.__matrix[row][col]
        else: raise StopIteration

    def info(self): return {"shape":self.__shape, "size":self.__size, "dtype":self.__dtype, "symbol":self.__symbol}

    # def __repr__(self): return f"<'Matrix' object at {hex(id(self))} size={self.__size} shape={self.__shape} symbol={self.__symbol}>"

    def __repr__(self):
        formatted_matrix = self.__formating()
        symbol_str = f"'{self.__symbol}'" if self.__symbol else None
        return f"Matrix({formatted_matrix}, dtype={self.__dtype.__name__}, symbol={symbol_str}, shape={self.__shape})"

    def __formating(self):
        if not self.__matrix or not self.__matrix[0]: return "[]"
        def format_element(element):
            if isinstance(element,(float,complex)): return f"{element:.5f}"
            return str(element)
        max_width = max(len(format_element(element)) for row in self.__matrix for element in row)
        rows = []
        for i, row in enumerate(self.__matrix):
            if len(row) > 7:
                row_str = ", ".join(f"{format_element(element):>{max_width}}" for element in row[:3])
                row_str += ", ..., " + ", ".join(f"{format_element(element):>{max_width}}" for element in row[-3:])
            else: row_str = ", ".join(f"{format_element(element):>{max_width}}" for element in row)
            rows.append(row_str)
        
        if len(rows) > 7:
            result = "[\n"
            result += ",\n".join(f"    [{row}]" for row in rows[:3])
            result += ",\n    ...,\n"
            result += ",\n".join(f"    [{row}]" for row in rows[-3:])
            result += "\n   ]"
        else:
            result = "[\n"
            result += ",\n".join(f"    [{row}]" for row in rows)
            result += "\n]"
        return result

    def __call__(self): return f"<'Matrix' object at {hex(id(self))} dtype={self.__dtype.__name__} size={self.__size} shape={self.__shape} symbol={self.__symbol}>"

    def call(self): return self()

    def __del__(self):
        if hasattr(self,"__symbol") and self.__symbol: del __mem__[self.__symbol]

    def assign_symbol(self, new_symbol:str): 
        if self.__symbol is None and new_symbol not in __mem__.keys():
            self.__symbol = new_symbol
            __mem__[self.__symbol] = self()
        else: raise ValueError(f"'{new_symbol}' already exists! try to use different symbol")
        return self.update_symbol(new_symbol)

    def update_symbol(self, new_symbol:str):
        if self.__symbol is None: return self.assign_symbol(new_symbol)
        else:
            del __mem__[self.__symbol]
            self.__symbol = new_symbol
            __mem__[self.__symbol] = self()

    def remove_symbol(self):
        if self.__symbol is not None:
            del __mem__[self.__symbol]
            self.__symbol = None

    def __getitem__(self, indices):
        if isinstance(indices,tuple) and len(indices) == 2:
            row,col = indices
            return self.__matrix[row][col]
        elif isinstance(indices,int):
            row = indices
            return self.__matrix[row]
        raise TypeError("Indexing should be either a tuple for 2D or an integer for 1D row assignment")

    def __setitem__(self, indices, value):
        if isinstance(indices,tuple) and len(indices) == 2:
            row,col = indices
            if not isinstance(value,str):
                self.__matrix[row][col] = value
                if isinstance(value,complex): self.astype(complex, inplace=True)
            else: raise TypeError("expected value to be an int or float")
            self.astype(self.__res_dtype__(value), inplace=True)
        elif isinstance(indices,int):
            row = indices
            if isinstance(value,list):
                if len(value) == self.__col:
                    self.__matrix[row] = value
                    if any(isinstance(v,complex) for v in value): self.astype(complex, inplace=True)
                    elif all(isinstance(v,bool) for v in value): self.astype(bool, inplace=True)
                    elif all(isinstance(v,int) for v in value): self.astype(int,inplace=True)
                    else: self.astype(float, inplace=True) 
                else: raise ValueError("number of elements in the row must be equal to the number of columns")
            else: raise TypeError("expected value to be a list")
        else: raise TypeError("Indexing should be either a tuple for 2D or an integer for 1D row assignment")

    @property
    def row(self): return self.__row

    @property
    def col(self): return self.__col
    
    @property
    def shape(self): return self.__shape
    
    @property
    def size(self): return self.__size

    @property
    def symbol(self): return self.__symbol

    @property
    def T(self): return Matrix([[self.__matrix[j][i] for j in range(self.__row)] for i in range(self.__col)], dtype=self.__dtype)

    @property
    def dtype(self): return self.__dtype.__name__

    @property
    def imag(self):
        if not self.__dtype == complex: return zeros((self.__row,self.__col))
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x.imag)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float)

    @property
    def real(self):
        if not self.__dtype == complex: return Matrix(self.__matrix)
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x.real)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float)

    @classmethod
    def from_numpy(cls, array, symbol:Optional[str]=None):
        if isinstance(array,np.ndarray): return cls(array.tolist(),symbol)
        raise TypeError("input should be a numpy array")
    
    @staticmethod
    def __exp(value: Union[int,float]): return 2.7182818284590452353602874713527 ** value

    def numpy(self): return np.array(self.__matrix)

    def __add__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] + other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] + other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __radd__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other + self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __sub__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] - other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] - other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rsub__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other - self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __mul__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] * other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] * other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rmul__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other * self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __matmul__(self, other):
        if not isinstance(other,Matrix): raise TypeError(f"multiplication is only supported between two matrices, not between a matrix and {type(other).__name__}")
        if self.__col != other.__row: raise ValueError(f"number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication")
        result_matrix = [[0 for _ in range(other.__col)] for _ in range(self.__row)]
        for i in range(self.__row):
            for j in range(other.__col):
                for k in range(self.__col): result_matrix[i][j] += self.__matrix[i][k] * other.__matrix[k][j]
        return Matrix(result_matrix, dtype=self.__res_dtype__(other))

    def __truediv__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] / other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] / other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rtruediv__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other / self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __floordiv__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] // other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] // other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rfloordiv__(self, other):
        new_mat = []
        for row in range(self.__row):
           buffer = []
           for x in range(self.__col): buffer.append(other // self.__matrix[row][x])
           new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __pow__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] ** other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] ** other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rpow__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other ** self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __mod__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] % other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=self.__res_dtype__(other))
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] % other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))

    def __rmod__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other % self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other))
    
    def __abs__(self):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x < 0: buffer.append(-x)
                else: buffer.append(x)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype)

    def __pos__(self): return Matrix(self.__matrix, dtype=self.__dtype)

    def __neg__(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(-self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype)

    def __and__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] & other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=int)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] & other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)
        
    def __rand__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other & self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __or__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] | other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=int)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] | other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __ror__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other | self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __invert__(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(~self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __xor__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] ^ other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=int)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] ^ other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __rxor__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other ^ self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __lshift__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] << other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=int)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] << other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)
    
    def __rlshift__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other << self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __rshift__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] >> other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=int)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] >> other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)
    
    def __rrshift__(self, other):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(other >> self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int)

    def __eq__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__matrix} != {other.__matrix}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] == other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool) 
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] == other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)

    def __ne__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] != other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] != other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)
        
    def __lt__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] < other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] < other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)
        
    def __gt__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] > other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] > other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)
        
    def __le__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] <= other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] <= other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)

    def __ge__(self, other):
        if isinstance(other,Matrix):
            if self.__shape != other.__shape: raise ValueError(f"the shape of the matrices must be the same, {self.__shape} != {other.__shape}")
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(self.__matrix[row][x] >= other.__matrix[row][x])
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=bool)
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] >= other)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool)

    def hstack(
            self,
            other:Matrix,
            symbol:Optional[str]=None
        ):
        if not isinstance(other,Matrix): raise TypeError(f"`{type(other).__name__}` can't be stacked with `Matrix`")
        if self.__row != other.__row: raise ValueError(f"number of rows must be equal in both matrices, ({self.__row} != {other.__row})")
        new_mat = []
        for row in range(self.__row):
            buffer = self.__matrix[row] + other.__matrix[row]
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(other), symbol=symbol)

    def vstack(
            self,
            other:Matrix,
            symbol:Optional[str]=None
        ):
        if not isinstance(other,Matrix): raise TypeError(f"`{type(other).__name__}` can't be stacked with `Matrix`")
        if self.__col != other.__col: raise ValueError(f"number of cols must be equal in both matrices, ({self.__col} != {other.__col})")
        new_mat = []
        for row in range(self.__row): new_mat.append(self.__matrix[row])
        for row in range(other.__row): new_mat.append(other.__matrix[row])
        return Matrix(new_mat, dtype=self.__res_dtype__(other), symbol=symbol)

    def append(
            self,
            array1d:List[Union[int,float,bool,complex]],
            axis:int=0,
            index:int=0
        ):
        if axis not in [None,0,1]: raise ValueError("axis must be None, 0 (row-wise) and 1 (column-wise)")
        if axis == 0:
            if len(array1d) != self.__col: raise ValueError(f"length of the row to append ({len(array1d)}) must match the number of columns ({self.__col})")
            self.__matrix.insert(index,array1d)
            self.__row += 1
        elif axis == 1:
            if len(array1d) != self.__row: raise ValueError(f"length of the column to append ({len(array1d)}) must match the number of rows ({self.__row})")
            for x in range(self.__row): self.__matrix.insert(index,array1d)
            self.__col += 1
        self.astype(dtype=self.__res_dtype__(array1d[0]), inplace=True)
        self.__shape = (self.__row,self.__col)
        self.__size = self.__row * self.__col
    
    def remove(
            self,axis:int=0,
            index:int=0
        ):
        if axis not in [None,0,1]: raise ValueError("axis must be None, 0 (row-wise) and 1 (column-wise)")
        if axis == 0:
            if index >= self.__row and index < 0: raise ValueError(f"row index {index} is out of bound for axis 0 with size {self.__row}")
            del self.__matrix[index]
            self.__row -= 1
        elif axis == 1:
            if index >= self.__col or index < 0: raise ValueError(f"column index {index} is out of bound for axis 1 with size {self.__col}")
            for row in self.__matrix: del row[index]
            self.__col -= 1
        self.__shape = (self.__row,self.__col)
        self.__size = self.__row * self.__col

    def replace(
            self,
            array1d:List[Union[int,float,bool,complex]],
            axis:int=0,
            index:int=0
        ):
        if axis not in [None,0,1]: raise ValueError("axis must be None, 0 (row-wise) and 1 (column-wise)")
        if axis == 0:
            if len(array1d) != self.__col: raise ValueError(f"length of the new row ({len(array1d)}) must match the number of columns ({self.__col})")
            if not 0 <= index < self.__row: raise IndexError(f"row index {index} is out of bounds")
            self.__matrix[index] = array1d
        elif axis == 1:
            if len(array1d) != self.__row: raise ValueError(f"length of the new column ({len(array1d)}) must match the number of rows ({self.__row})")
            if not 0 <= index < self.__col: raise IndexError(f"col index {index} is out of bounds")
            for x in range(self.__row): self.__matrix[x][index] = array1d[x]
        self.astype(self.__res_dtype__(array1d[0]), inplace=True)

    def cumsum(
            self,
            axis:Union[int,None]=None,
            symbol:Optional[str]=None
        ):
        if axis not in [None,0,1]: raise ValueError("axis must be None, 0 (row-wise) and 1 (column-wise)")
        if axis is None:
            flattend = [x for row in self.__matrix for x in row]
            cumsum_flattend = [sum(flattend[:i+1]) for i in range(len(flattend))]
            result_matrix = [cumsum_flattend[i:i + self.__col] for i in range(0,len(cumsum_flattend),self.__col)]
        elif axis == 0:
            result_matrix = []
            for col in range(self.__col):
                col_cumsum = []
                cumsum_ = 0
                for row in range(self.__row):
                    cumsum_ += self.__matrix[row][col]
                    col_cumsum.append(cumsum_)
                result_matrix.append(col_cumsum)
            result_matrix = list(map(list,zip(*result_matrix)))
        elif axis == 1:
            result_matrix = []
            for row in self.__matrix:
                row_cumsum = []
                cumulative_sum = 0
                for element in row:
                    cumulative_sum += element
                    row_cumsum.append(cumulative_sum)
                result_matrix.append(row_cumsum)
        return Matrix(result_matrix, dtype=self.__dtype, symbol=symbol)
    
    def cumprod(
            self,
            axis:Union[int,None]=None,
            symbol:Optional[str]=None
        ):
        if axis not in [None,0,1]: raise ValueError("axis must be None, 0 (row-wise) and 1 (column-wise)")
        if axis is None:
            flattend = [x for row in self.__matrix for x in row]
            cumprod_flattend = [flattend[0]]
            for i in range(1,len(flattend)): cumprod_flattend.append(cumprod_flattend[-1] * flattend[i])
            result_matrix = [cumprod_flattend[i:i + self.__col] for i in range(0,len(cumprod_flattend),self.__col)]
        elif axis == 0:
            result_matrix = []
            for col in range(self.__col):
                col_cumprod = []
                cumulative_product = 1
                for row in range(self.__row):
                    cumulative_product *= self.__matrix[row][col]
                    col_cumprod.append(cumulative_product)
                result_matrix.append(col_cumprod)
            result_matrix = list(map(list,zip(*result_matrix)))
        elif axis == 1:
            result_matrix = []
            for row in self.__matrix:
                row_cumprod = []
                cumulative_product = 1
                for element in row:
                    cumulative_product *= element
                    row_cumprod.append(cumulative_product)
                result_matrix.append(row_cumprod)
        return Matrix(result_matrix, dtype=self.__dtype, symbol=symbol)
        
    def gradient(self, axis:int=0, symbol:Optional[str]=None):
        if axis not in [0,1]: raise ValueError("axis must be 0 (row-wise) and 1 (column-wise)")
        grad = []
        if axis == 0:
            for row in range(self.__row):
                grad_row = []
                for x in range(self.__col):
                    if row == 0: grad_val = self.__matrix[row][x] - self.__matrix[row][x]
                    elif row == self.__row - 1: grad_val = self.__matrix[row][x] - self.__matrix[row - 1][x]
                    else: grad_val = (self.__matrix[row + 1][x] - self.__matrix[row - 1][x]) / 2
                    grad_row.append(grad_val)
                grad.append(grad_row)
        elif axis == 1:
            for row in range(self.__row):
                grad_row = []
                for x in range(self.__col):
                    if x == 0: grad_val = self.__matrix[row][x + 1] - self.__matrix[row][x]
                    elif x == self.__col - 1: grad_val = self.__matrix[row][x] - self.__matrix[row][x - 1]
                    else: grad_val = (self.__matrix[row][x + 1] - self.__matrix[row][x - 1]) / 2
                    grad_row.append(grad_val)
                grad.append(grad_row)
        return Matrix(grad, dtype=self.__dtype, symbol=symbol)

    def sigmoid(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append((1 + self.__exp(-self.__matrix[row][x])) ** -1)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def hard_sigmoid(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(min(0,max(1,(self.__matrix[row][x] + 1)/2)))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def hard_shrink(self, lambda_:Union[float,int]=0.5, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                if self.__matrix[row][x] > lambda_: buffer.append(self.__matrix[row][x])
                elif self.__matrix[row][x] < -lambda_: buffer.append(self.__matrix[row][x])
                else: buffer.append(0)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def hard_tanh(
            self,
            min_value:Union[int,float]=-0.5,
            max_val:Union[int,float]=0.5,
            inplace:bool=False,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                if self.__matrix[row][x] > max_val: buffer.append(max_val)
                elif self.__matrix[row][x] < min_value: buffer.append(min_value)
                else: buffer.append(x)
            new_mat.append(buffer)
        if inplace: self.__matrix = new_mat; return self
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def hard_swish(self, inplace:bool=False, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                if self.__matrix[row][x] <= -3: buffer.append(0)
                elif self.__matrix[row][x] >= 3: buffer.append(self.__matrix[row][x])
                else: buffer.append(self.__matrix[row][x] * (self.__matrix + 3) / 6)
            new_mat.append(buffer)
        if inplace: self.__matrix = new_mat; return self
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def relu(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"relu can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(max(0,self.__matrix[row][x]))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def leaky_relu(
            self,
            negative_slope:Union[int,float],
            inplace:bool=False,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                if self.__matrix[row][x] >= 0: buffer.append(self.__matrix)
                else: buffer.append(negative_slope * self.__matrix[row][x])
            new_mat.append(buffer)
        if inplace: self.__matrix = new_mat; return self
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def log_sigmoid(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.log10((1 + np.exp(-x))**-1))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def prelu(self, a:Union[int,float], symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError("prelu can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(a,(int,float)): raise TypeError(f"Invalid 'a' dtype, expected either int or float but got '{type(a).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x >= 0: buffer.append(x)
                else: buffer.append(a*x)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float, symbol=symbol)

    def relu6(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"relu6 can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(min(max(0,x),6))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def elu(self, alpha:Union[int,float]=1.0, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"elu can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(alpha,(int,float)): raise TypeError(f"Invalid alpha dtype, expected either int or float but got '{type(alpha).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x > 0: buffer.append(x)
                else: buffer.append(alpha * (np.exp(x) - 1))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def rrelu(self, upper:Union[int,float]=1/8, lower:Union[int,float]=1/3, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"rrelu can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        a = (lower + upper) / 2
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x >= 0: buffer.append(x)
                else: buffer.append(a * x)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def selu(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"selu can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(scale * (max(0,x) + min(0,alpha * (np.exp(x) - 1))))
            new_mat.append(buffer)
        return Matrix(new_mat, symbol=symbol)

    def celu(self, alpha:Union[int,float]=1.0, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"celu can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(alpha,(int,float)): raise TypeError(f"Invalid alpha dtype, expected either int or float but got '{type(alpha).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(max(0,x) + min(0,alpha * (np.exp(x/alpha) - 1)))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def gelu(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"gelu can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                value = 0.5 * x + (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x ** 3)))
                buffer.append(value)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def silu(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"silu can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                value = x * (1 + np.exp(-x) ** -1)
                buffer.append(value)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def mish(self, beta:Union[int,float]=1.0, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"mish can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(beta,(int,float)): raise TypeError(f"Invalid beta dtype, expected either int or float but got '{type(alpha).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                sp = beta ** -1 * np.log(1 + np.exp(beta * x))
                value = x * np.tanh(sp)
                buffer.append(value)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def softshrink(self, lambda_:Union[int,float]=0.5, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"softshrink can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(lambda_,(int,float)): raise TypeError(f"Invalid lambda dtype, expected either int or float but got '{type(lambda_).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x > lambda_: buffer.append(x - lambda_)
                elif x < -lambda_: buffer.append(x + lambda_)
                else: buffer.append(0)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def softsign(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"softsign can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x / (1 + np.abs(x)))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def tanhshrink(self, symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"tanhshrink can't be implemented on 'complex' or 'bool' dtype matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x - np.tanh(x))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def threshold(self, value:Union[int,float], symbol:Optional[str]=None):
        if self.__dtype == complex or self.__dtype == bool: raise TypeError(f"threshold can't be implemented on 'complex' or 'bool' dtype matrices")
        if not isinstance(value,(int,float)): raise TypeError(f"invalid value dtype, expected either int or float but got '{type(valye).__name__}'")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x > value: buffer.append(x)
                else: buffer.append(value)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def logsoftmax(self, axis:int=0, symbol:Optional[str]=None):
        if axis not in [0,1]: raise ValueError("axis must be 0(row-wise) or 1(column-wise)")
        logsoftmax_matrix = []
        if axis == 0:
            for x in range(self.__col):
                column = [self.__matrix[y][x] for y in range(self.__row)]
                max_val = max(column)
                shifted_col = [val - max_val for val in column]
                exp_col = [self.__exp(val) for val in shifted_col]
                sum_exp = sum(exp_col)
                log_sum_exp = self.__log(sum_exp)
                logsoftmax_column = [val - max_val - log_sum_exp for val in column]
                logsoftmax_matrix.append(logsoftmax_column)
            logsoftmax_matrix = list(map(list,zip(*logsoftmax_matrix)))
        elif axis == 1:
            for i in range(self.__row):
                row = self.__matrix[i]
                max_val = max(row)
                shifted_row = [val - max_val for val in row]
                exp_row = [self.__exp(val) for val in shifted_row]
                sum_exp = sum(exp_row)
                log_sum_exp = self.__log(sum_exp)
                logsoftmax_row = [val - max_val - log_sum_exp for val in row]
                logsoftmax_matrix.append(logsoftmax_row)
            return Matrix(logsoftmax_matrix, dtype=self.__dtype, symbol=symbol)

    def softplus(self, beta:Union[int,float]=1.0, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(beta ** -1 * np.log(1 + np.exp(beta * x)))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def softmax(self, axis:int=0, symbol:Optional[str]=None):
        if axis not in [0,1]: raise ValueError("axis must be 0 (row-wise) and 1 (column-wise)")
        softmax_matrix = []
        if axis == 0:
            for x in range(self.__col):
                column = [self.__matrix[y][x] for y in range(self.__row)]
                max_val = max(column)
                shifted_column = [val - max_val for val in column]
                exp_column = [self.__exp(i) for i in shifted_column]
                sum_exp = sum(exp_column)
                softmax_column = [exp_val / sum_exp for exp_val in exp_column]
                softmax_matrix.append(softmax_column)
            softmax_matrix = list(map(list,zip(*softmax_matrix)))
        elif axis == 1:
            for i in range(self.__row):
                row = self.__matrix[i]
                max_val = max(row)
                shifted_row = [val - max_val for val in row]
                exp_row = [self.__exp(x) for x in shifted_row]
                sum_exp = sum(exp_row)
                softmax_row = [exp_val / sum_exp for exp_val in exp_row]
                softmax_matrix.append(softmax_row)
        return Matrix(softmax_matrix, dtype=self.__dtype, symbol=symbol)

    def softmin(self, axis:int=0, symbol:Optional[str]=None):
        if axis not in [0, 1]: raise ValueError("axis must be 0 (row-wise) or 1 (column-wise)")
        softmin_matrix = []
        if axis == 0:
            for x in range(self.__col):
                column = [self.__matrix[y][x] for y in range(self.__row)]
                max_neg = max(-val for val in column)
                shifted_column = [-val - max_neg for val in column]
                exp_column = [self.__exp(i) for i in shifted_column]
                sum_exp = sum(exp_column)
                softmin_column = [exp_val / sum_exp for exp_val in exp_column]
                softmin_matrix.append(softmin_column)
            softmin_matrix = list(map(list, zip(*softmin_matrix)))
        elif axis == 1:
            for i in range(self.__row):
                row = self.__matrix[i]
                max_neg = max(-val for val in row)
                shifted_row = [-val - max_neg for val in row]
                exp_row = [self.__exp(i) for i in shifted_row]
                sum_exp = sum(exp_row)
                softmin_row = [exp_val / sum_exp for exp_val in exp_row]
                softmin_matrix.append(softmin_row)
        return Matrix(softmin_matrix, dtype=self.__dtype, symbol=symbol)

    def conv2d(self, kernel:Matrix, padding:int=0):
        padded_matrix = self.__pad_matrix(padding)
        kernel_rows, kernel_cols = kernel.__shape
        output_rows = self.__row - kernel_rows + 1 + 2 * padding
        output_cols = self.__col - kernel_cols + 1 + 2 * padding
        output = [[0 for _ in range(output_cols)] for _ in range(output_rows)]
        for i in range(output_rows):
            for j in range(output_cols):
                sub_matrix = [row[j:j+kernel_cols] for row in padded_matrix[i:i+kernel_rows]]
                output[i][j] = self.__apply_kernel(sub_matrix, kernel)
        return Matrix(output)

    def __pad_matrix(self, padding):
        if padding == 0: return self.__matrix
        padded = [[0] * (self.__col + 2 * padding) for _ in range(self.__row + 2 * padding)]
        for row in range(self.__row):
            for x in range(self.__col): padded[row + padding][x + padding] = self.__matrix[row][x]
        return Matrix(padded)

    def __apply_kernel(self, sub_matrix, kernel):
        result = 0
        for row in range(len(sub_matrix)):
            for x in range(len(sub_matrix[0])): result += sub_matrix[row][x] * kernel.__matrix[row][x]
        return result

    def reciprocate(self):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                buffer.append(1 / self.__matrix[row][x])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float)
    
    def isnan(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                if isinstance(self.__matrix[row][x],float) and self.__matrix[row][x] != self.__matrix[row][x]: buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    
    def isinf(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x == float("inf") or x == -float("inf"): buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    
    def isposinf(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x == float("inf"): buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)

    def isneginf(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x == -float("inf"): buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)
    
    def isfinite(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x == float("inf") or x == -float("inf") or x != x: buffer.append(False)
                else: buffer.append(True)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)

    def isreal(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x.imag == 0: buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)

    def isimag(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                if x.imag != 0: buffer.append(True)
                else: buffer.append(False)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=bool, symbol=symbol)

    def isnull(self):
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] == 0: continue
                else: return False
        return True
    
    def countnan(self):
        count = 0
        for row in range(self.__row):
            for x in range(self.__col):
                if isinstance(self.__matrix[row][x],float) and self.__matrix[row][x] != self.__matrix[row][x]: count += 1
        return count

    def matpow(self, n:int):
        if not isinstance(n,int): raise TypeError(f"given base must be an integer not `{type(n).__name__}`")
        elif n < 0: raise ValueError("`n` must be greater than 0")
        result = Matrix([[1 if i == j else 0 for j in range(self.__col)] for i in range(self.__row)])
        base = self
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
        return result

    def vecmul(self, vector:List[Union[int,float,bool,complex]]):
        if self.__col != len(vector): raise ValueError(f"length of a vector must be equal to the number of columns of that matrix")
        result = [0] * self.__row
        for row in range(self.__row):
            for x in range(self.__col): result[row] += self.__matrix[row][x] * vector[x]
        return result

    def conj(self, symbol:Optional[str]=None):
        if not self.__dtype == complex: return Matrix(self.__matrix, dtype=self.__dtype, symbol=symbol)
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x.conjugate())
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=complex, symbol=symbol)
    
    def mag(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(np.sqrt(x.real**2 + x.imag**2))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float, symbol=symbol)

    def arg(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(np.arctan(x.imag/x.real))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float, symbol=symbol)

    def sgn(self, symbol:Optional[str]=None):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row:
                mag = np.sqrt(x.real**2 + x.imag**2)
                if mag == 0: buffer.append(0)
                else: buffer.append(x / mag)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=complex, symbol=symbol)

    def pos(self): return +self
    def neg(self): return -self

    def add(self, other:List[Union[Matrix,int,float,complex]]): return self + other
    def sub(self, other:List[Union[Matrix,int,float,complex]]): return self - other
    def mul(self, other:List[Union[Matrix,int,float,complex]]): return self * other
    def truediv(self, other:List[Union[Matrix,int,float,complex]]): return self / other
    def floordiv(self, other:List[Union[Matrix,int,float,complex]]): return self // other
    def pow(self, other:list[Union[Matrix,int,float,complex]]): return self ** other
    def mod(self, other:List[Union[Matrix,int,float,complex]]): return self % other
    def matmul(self, other:Matrix): return self @ other
    def abs(self): return self.__abs__()
    def absolute(self): return self.__abs__()

    def bitwise_and(self, other:List[Union[Matrix,int]]): return self & other
    def bitwise_nand(self, other:List[Union[Matrix,int]]): return ~(self & other)
    def bitwise_or(self, other:List[Union[Matrix,int]]): return self | other
    def bitwise_nor(self, other:List[Union[Matrix,int]]): return ~(self | other)
    def bitwise_invert(self): return ~self
    def bitwise_xor(self, other:List[Union[Matrix,int]]): return self ^ other
    def bitwise_xnor(self, other:List[Union[Matrix,int]]): return ~(self ^ other)
    def bitwise_left_shift(self, other:List[Union[Matrix,int]]): return self << other
    def bitwise_right_shift(self, other:List[Union[Matrix,int]]): return self >> other

    def eq(self, other:List[Union[Matrix,int,float,complex]]): return  self == other
    def ne(self, other:List[Union[Matrix,int,float,complex]]): return self != other
    def gt(self, other:List[Union[Matrix,int,float]]): return self > other
    def lt(self, other:List[Union[Matrix,int,float]]): return self < other
    def ge(self, other:List[Union[Matrix,int,float]]): return self >= other
    def le(self, other:List[Union[Matrix,int,float]]): return self <= other

    def rand_like(
            self,
            seed:Optional[int]=None,
            symbol:Optional[str]=None
        ):
        new_mat = []
        if seed is not None: np.random.seed(seed)
        for _ in range(self.__row):
            buffer = []
            for _ in range(self.__col):
                if self.__dtype == int: buffer.append(np.random.randint(0,100))
                elif self.__dtype == float: buffer.append(np.random.rand())
                elif self.__dtype == complex:
                    real_part = np.random.rand()
                    imag_part = np.random.rand()
                    buffer.append(complex(real_part,imag_part))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def zeros_like(
            self,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for col in range(self.__col): buffer.append(0)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    
    def ones_like(
            self,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(1)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)
    
    def fill_like(
            self,
            value:Union[int,float,bool,complex],
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(value)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=type(value), symbol=symbol)

    def max_pooling(
            self,
            kernel_size:Tuple[int,int],
            stride:Tuple[int,int]=(1,1),
            symbol:Optional[str]=None
        ):
        rows, cols = self.shape
        k_rows, k_cols = kernel_size
        s_rows, s_cols = stride
        pooled_matrix = []
        for i in range(0,rows - k_rows + 1,s_rows):
            row = []
            for j in range(0,cols - k_cols + 1,s_cols):
                max_value = max(
                    self.__matrix[i + m][j + n]
                    for m in range(k_rows)
                    for n in range(k_cols)
                )
                row.append(max_value)
            pooled_matrix.append(row)
        return Matrix(pooled_matrix, dtype=self.__dtype, symbol=symbol)

    def min_pooling(
            self,
            kernel_size:Tuple[int,int],
            stride:Tuple[int,int]=(1,1),
            symbol:Optional[str]=None
        ):
        rows, cols = self.shape
        k_rows, k_cols = kernel_size
        s_rows, s_cols = stride
        pooled_matrix = []
        for i in range(0,rows - k_rows + 1,s_rows):
            row = []
            for j in range(0,cols - k_cols + 1,s_cols):
                min_value = min(
                    self.__matrix[i + m][j + n]
                    for m in range(k_rows)
                    for n in range(k_cols)
                )
                row.append(min_value)
            pooled_matrix.append(row)
        return Matrix(pooled_matrix, dtype=self.__dtype, symbol=symbol)

    def avg_pooling(
            self,
            kernel_size:Tuple[int,int],
            stride:Tuple[int,int]=(1,1),
            symbol:Optional[str]=None
        ):
        rows, cols = self.shape
        k_rows, k_cols = kernel_size
        s_rows, s_cols = stride
        pooled_matrix = []
        for i in range(0,rows - k_rows + 1,s_rows):
            row = []
            for j in range(0,cols - k_cols + 1,s_cols):
                sum_value = sum(
                    self.__matrix[i + m][j + n]
                    for m in range(k_rows)
                    for n in range(k_cols)
                )
                avg_value = sum_value / (k_rows * k_cols)
                row.append(avg_value)
            pooled_matrix.append(row)
        return Matrix(pooled_matrix, dtype=self.__dtype, symbol=symbol)

    def astype(
            self,
            dtype:Type,
            inplace:bool=False
        ):
        if inplace:
            for row in range(self.__row):
                for x in range(self.__col): self.__matrix[row][x] = dtype(self.__matrix[row][x])
            self.__dtype = dtype
        elif not inplace:
            new_mat = []
            for row in range(self.__row):
                buffer = []
                for x in range(self.__col): buffer.append(dtype(self.__matrix[row][x]))
                new_mat.append(buffer)
            return Matrix(new_mat, dtype=dtype)
        else: raise ValueError("invalid argument for `dtype` expected from `True` or `False`")

    def copy(self, symbol:Optional[str]=None): return Matrix([row[:] for row in self.__matrix], dtype=self.__dtype, symbol=symbol)
    def cp(self, symbol:Optional[str]=None): return Matrix([row[:] for row in self.__matrix], dtype=self.__dtype, symbol=symbol)

    def reverse(self, mode:int=0, symbol:Optional[str]=None):
        if mode == 0: return Matrix([rows[::-1] for rows in self.__matrix], dtype=self.__dtype, symbol=symbol)
        elif mode == 1: return Matrix(self.__matrix[::-1], dtype=self.__dtype, symbol=symbol)
        elif mode == 2: return Matrix([row[::-1] for row in self.__matrix[::-1]], dtype=self.__dtype, symbol=symbol)
        raise ValueError(f"expected mode values are 0, 1 or 2 but expected {mode}")

    def clip(
            self,
            min_value:Union[int,float],
            max_value:Union[int,float],
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(min(max(x,min_value),max_value))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def clamp(
            self,
            min_value:Union[int,float],
            max_value:Union[int,float]
        ): return self.clip(min_value=min_value, max_value=max_value)

    def get(
            self,
            row:int,
            col:None|int=None
        ):
        if col is None: return self.__matrix[row]
        else: return self.__matrix[row][col]

    def set(
            self,
            row:int,
            value:List[Union[int,float,bool,complex]] | int | float,
            col:None|int=None
        ):
        if col is None:
            if isinstance(value, list):
                if len(value) != self.__col: raise ValueError(f"length of value list ({len(value)}) does not match number of columns ({self.__col})")
                self.__matrix[row] = value
                self.astype(dtype=self.__res_dtype__(value[0]), inplace=True)
            else: raise TypeError(f"expected list for row setting, got `{type(value).__name__}`")
        else:
            if not (0 <= col < self.__col): raise IndexError(f"column index {col} out of range (0 to {self.__col-1})")
            self.__matrix[row][col] = value
            self.astype(dtype=self.__res_dtype__(value), inplace=True)

    def concat(
            self,
            other:Union[Matrix,List],
            axis:int=0,
            symbol:Optional[str]=None
        ):
        if isinstance(other,Matrix):
            if axis == 0:
                if self.__col != other.__col: raise ValueError("Matrices must have the same number of columns to concatenate vertically")
                new_matrix = self.__matrix + other.__matrix
            elif axis == 1:
                if self.__row != other.__row: raise ValueError("Matrices must have the same number of rows to concatenate horizontally")
                new_matrix = [self.__matrix[row] + other.__matrix[row] for row in range(self.__row)]
            else: raise ValueError("Axis must be 0 (vertical) or 1 (horizontal)")
        elif isinstance(other,list):
            if axis == 0:
                if len(other) != self.__col: raise ValueError("The list must have the same number of elements as there are columns in the matrix to concatenate vertically")
                new_matrix = self.__matrix + [other]
            elif axis == 1:
                if len(other) != self.__row: raise ValueError("the list must have the same number of elements as there are rows in the matrix to concatenate horizontally")
                new_matrix = [self.__matrix[row] + [other[row]] for row in range(self.__row)]
            else: raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")
        else: raise TypeError(f"couldn't add `{type(other).__name__}` and `Matrix`, other must be `list` or `Matrix`")
        if isinstance(other,Matrix): return Matrix(new_matrix, dtype=self.__res_dtype__(other), symbol=symbol)
        return Matrix(new_matrix, dtype=self.__res_dtype__(other[0]), symbol=symbol)

    def to_json(self, file_path:str):
        if os.path.exists(file_path):
            raise FileExistsError(f"file with name `{file_path}` already exist!")
        with open(file_path,"w") as file:
            file.write("{\n")
            for row in range(self.__row):
                if row == self.__row - 1: file.write(f'   "{row}": {self.__matrix[row]}\n')
                else: file.write(f'   "{row}": {self.__matrix[row]},\n')
            file.write("}\n")
            file.close()

    def to_csv(self, file_path:str):
        if os.path.exists(file_path): raise FileExistsError(f"file with name `{file_path}` already exist!")
        with open(file_path,"w") as file:
            for row in range(self.__row):
                for x in range(self.__col):
                    if x == self.__col - 1: file.write(f"{self.__matrix[row][x]}")
                    else: file.write(f"{self.__matrix[row][x]},")
                file.write("\n")
            file.close()

    def to_dict(self):
        result = {}
        for row in range(self.__row): result[f"{row}"] = self.__matrix[row]
        return result

    def to_list(self, keepdim:bool=True):
        if keepdim: return self.__matrix
        return self.squeeze()

    def squeeze(self):
        result = []
        for row in self.__matrix:
            for x in row: result.append(x)
        return result

    def mean(
            self,
            axis=None,
            symbol:Optional[str]=None
        ):
        if axis is None:
            if symbol is not None: warnings.warn("Warning: the 'symbol' parameter is ignored when the 'axis' is None",UserWarning)
            return sum([x for row in self.__matrix for x in row]) / self.__size
        if axis == 1:
            new_mat = []
            for row in range(self.__row):
                total = 0
                for x in range(self.__col): total += self.__matrix[row][x]
                new_mat.append([total / self.__col])
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        elif axis == 0:
            new_mat = []
            for x in range(self.__col):
                total = 0
                for row in range(self.__row): total += self.__matrix[row][x]
                new_mat.append([total / self.__row])
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        else: raise ValueError("invalid axis given!")

    def var(
            self,
            axis=None,
            sample=True,
            symbol:Optional[str]=None
        ):
        if axis is None:
            if symbol is not None: warnings.warn("Warning: the 'symbol' parameter is ignored when 'axis' is None.", UserWarning)
            total_elements = sum(len(row) for row in self.__matrix)
            mean_value = self.mean()
            variance_sum = sum((element - mean_value) ** 2 for row in self.__matrix for element in row)
            if sample and total_elements > 1: return variance_sum / (total_elements - 1)
            else: return variance_sum / total_elements
        elif axis == 1:
            mean_matrix = self.mean(axis=1).to_list()
            new_mat = []
            for row in range(self.__row):
                total = 0
                for x in range(self.__col): total += (self.__matrix[row][x] - mean_matrix[row][0]) ** 2.0
                if sample and self.__col > 1: new_mat.append([total / (self.__col - 1)])
                else: new_mat.append([total / self.__col])
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        elif axis == 0:
            mean_matrix = self.mean(axis=0).to_list()
            new_mat = []
            for x in range(self.__col):
                total = 0
                for row in range(self.__row): total += (self.__matrix[row][x] - mean_matrix[x][0]) ** 2.0
                if sample and self.__row > 1: new_mat.append([total / (self.__row - 1)])
                else: new_mat.append([total / self.__row])
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        else: raise ValueError("Invalid axis given!")

    def std(
            self,
            axis=None,
            sample=True,
            symbol:Optional[str]=None
        ):
        if axis is None: return Matrix([[self.var()]]).sqrt()[0][0]
        return self.var(axis = axis,sample = sample,symbol = symbol).sqrt()

    def median(
            self,
            axis:int=1,
            symbol:Optional[str]=None
        ):
        if self.__dtype == complex: raise TypeError("median calculation is not supported for complex datatypes")
        if axis is None:
            if symbol is not None: warnings.warn("Warning: the 'symbol' parameter is ignored when 'axis' is None",UserWarning)
            flattened = [element for row in self.__matrix for element in row]
            sorted_flattened = sorted(flattened)
            mid = len(sorted_flattened) // 2
            if len(sorted_flattened) % 2 == 0: median_value = (sorted_flattened[mid - 1] + sorted_flattened[mid]) / 2.0
            else: median_value = sorted_flattened[mid]
            return median_value
        if axis == 1:
            new_mat = []
            for row in range(self.__row):
                sorted_row = sorted(self.__matrix[row])
                mid = self.__col // 2
                if self.__col % 2 == 0: median_value = (sorted_row[mid - 1] + sorted_row[mid]) / 2.0
                else: median_value = sorted_row[mid]
                new_mat.append([median_value])
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        elif axis == 0:
            new_mat = []
            for x in range(self.__col):
                col_values = [self.__matrix[row][x] for row in range(self.__row)]
                sorted_col = sorted(col_values)
                mid = self.__row // 2
                if self.__row % 2 == 0: median_value = (sorted_col[mid - 1] + sorted_col[mid]) / 2.0
                else: median_value = sorted_col[mid]
                new_mat.append(median_value)
            return Matrix([new_mat], dtype=self.__dtype, symbol=symbol)
        else:
            raise ValueError("invalid axis given!")

    def max(self, axis:None|int=None):
        if self.__dtype == complex: raise TypeError("max calculation is not supported for complex datatypes")
        if axis is None:
            result = -float("inf")
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result < value: result = value
            return result
        elif axis == 0:
            result = [-float("inf")] * self.__col
            for x in range(self.__col):
                for row in range(self.__row):
                    value = self.__matrix[row][x]
                    if result[x] < value: result[x] = value
            return result
        elif axis == 1:
            result = [-float("inf")] * self.__row
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result[row] < value: result[row] = value
            return result
        else:
            raise ValueError("invalid argument for axis it can either be None, 0 or 1")
    
    def min(self, axis:None|int=None):
        if self.__dtype == complex: raise TypeError("min calculation is not supported for complex datatypes")
        if axis is None:
            result = float("inf")
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result > value: result = value
            return result
        elif axis == 0:
            result = [float("inf")] * self.__col
            for x in range(self.__col):
                for row in range(self.__row):
                    value = self.__matrix[row][x]
                    if result[x] > value: result[x] = value
            return result
        elif axis == 1:
            result = [float("inf")] * self.__row
            for row in range(self.__row):
                for x in range(self.__col):
                    value = self.__matrix[row][x]
                    if result[row] > value: result[row] = value
            return result
        else:
            raise ValueError("invalid argument for axis it can either be None, 0 or 1")

    def argmax(self):
        value = -float("inf")
        for row in self.__matrix:
            for x in row:
                if value < x: value = x
        return value

    def argmin(self):
        value = float("inf")
        for row in self.__matrix:
            for x in row:
                if value > x: value = x
        return value

    def sort(
            self,
            axis=None,
            symbol:Optional[str]=None
        ):
        if self.__dtype == complex: raise TypeError("matrix with complex datatype cannot be sorted")
        if axis is None:
            if symbol is not None: warnings.warn("Warning: the 'symbol' parameter is ignored when 'axis' is None",UserWarning)
            return from1d(sorted(self.flatten().to_list()[0]),self.__shape,symbol)
        elif axis == 1:
            new_mat = [sorted(row) for row in self.__matrix]
            return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)
        elif axis == 0:
            new_mat = [[self.__matrix[row][col] for row in range(self.__row)] for col in range(self.__col)]
            sorted_mat = [sorted(col) for col in new_mat]
            transposed_sorted_mat = [[sorted_mat[col][row] for col in range(self.__col)] for row in range(self.__row)]
            return Matrix(transposed_sorted_mat, dtype=self.__dtype, symbol=symbol)
        raise ValueError("invalid axis given!")

    def sqrt(
            self,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] ** 0.5)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def cbrt(
            self,
            symbol:Optional[str]=None
        ):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] ** (1/3))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__dtype, symbol=symbol)

    def floor(
            self,
            symbol:Optional[str]=None
        ):
        if self.__dtype == complex: raise TypeError("matrix with complex datatype cannot be calculate the floor")
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                number = self[row][x]
                if number >= 0: buffer.append(int(number))
                else:
                    if int(number) == number: buffer.append(number)
                    else: buffer.append(int(number) - 1)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)

    def ceil(
            self,
            symbol:Optional[str]=None
        ):
        if self.__dtype == complex: raise TypeError("matrix with complex datatype cannot be calculate the ceil")
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col):
                number = self[row][x]
                if number == int(number): buffer.append(int(number))
                else: buffer.append(int(number) + 1 if number > 0 else int(number))
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=int, symbol=symbol)

    def scale(self, scalar:Union[int,float,bool,complex], symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(self.__matrix[row][x] * scalar)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=self.__res_dtype__(scalar), symbol=symbol)

    def log(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(float(np.log10(self.__matrix[row][x])))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def ln(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(float(np.log(self.__matrix[row][x])))
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def rad2deg(self, symbol:Optional[str]=None):
        if self.__dtype == complex: raise TypeError("rad2deg is not supported for complex matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x * 180/np.pi)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float, symbol=symbol)

    def deg2rad(self, symbol:Optional[str]=None):
        if self.__dtype == complex: raise TypeError("rad2deg is not supported for complex matrices")
        new_mat = []
        for row in self.__matrix:
            buffer = []
            for x in row: buffer.append(x * np.pi/180)
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=float, symbol=symbol)

    def sin(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.sin(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def cos(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.cos(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def tan(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.tan(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def sec(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.cos(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def cosec(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.sin(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def cot(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.tan(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def asin(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arcsin(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = float
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def sinh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.sinh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def cosh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__row): buffer.append(np.cosh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def tanh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.tanh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)
    
    def sech(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.cosh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = complex
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)
    
    def cosech(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.sinh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)
    
    def coth(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.tanh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def asinh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arcsinh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)
    def acosh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arccosh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def atanh(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arctanh(self.__matrix[row][x]))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def acosech(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arcsinh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def asech(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arccosh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def acoth(self, symbol:Optional[str]=None):
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(np.arctanh(self.__matrix[row][x] ** -1))
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def exp(self, symbol:Optional[str]=None):
        E = 2.7182818284590452353602874713527
        new_mat = []
        for row in range(self.__row):
            buffer = []
            for x in range(self.__col): buffer.append(E ** self.__matrix[row][x])
            new_mat.append(buffer)
        if self.__dtype == complex: result_dtype = complex
        else: result_dtype = float
        return Matrix(new_mat, dtype=result_dtype, symbol=symbol)

    def eigen(self, symbol:Optional[str]=None):
        if not self.is_square(): raise ValueError("eigen values and its vectors are defined only for square matrices")
        value,vec = np.linalg.eig(self.numpy())
        return value,Matrix(vec.tolist(), dtype=self.__dtype, symbol=symbol)

    def cofactor(self, i, j, symbol:Optional[str]=None):
        sub_matrix = [row[:j] + row[j + 1:] for row in (self.__matrix[:i] + self.__matrix[i + 1:])]
        sign = (-1) ** (i + j)
        return sign * Matrix(sub_matrix, dtype=self.__dtype, symbol=symbol).det()

    def row_matrix(self): return self.__col == 1

    def column_matrix(self): return self.__row == 1
    
    def minor(self, i, j, symbol:Optional[str]=None):
        sub_matrix = [row[:j] + row[j+1:] for row in (self.__matrix[:i] + self.__matrix[i+1:])]
        return Matrix(sub_matrix, dtype=self.__dtype, symbol=symbol).det()

    def det(self):
        if not self.is_square(): raise ValueError("determinant is defined only for square matrices")
        if self.__row == 1: return self.__matrix[0][0]
        if self.__row == 2: return self.__matrix[0][0] * self.__matrix[1][1] - self.__matrix[0][1] * self.__matrix[1][0]
        det = 0
        for j in range(self.__col): det += self.__matrix[0][j] * self.cofactor(0,j)
        return det

    def inverse(self):
        det = self.det()
        if det == 0: raise ZeroDivisionError(f"inverse of a matrix with 'zero' determinant doesn't exist!")
        return self.adjoint().scale(1/det)
    
    def is_invertible(self):
        if not self.is_square(): raise ValueError("matrix must have equal number of rows and columns")
        return self.det() != 0

    def is_symmetric(self):
        if self.__row != self.__col: return False
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] != self.__matrix[x][row]: return False
        return True

    def is_skew_symmetric(self):
        if self.__row != self.__col: return False
        for i in range(self.__row):
            for j in range(self.__col):
                if self.__matrix[i][j] != -self.__matrix[j][i]: return False
        return True

    def is_singular(self):
        if not self.is_square(): raise ValueError("matrix must have equal number of rows and columns")
        return self.det() == 0 

    def is_square(self): return self.__row == self.__col

    def is_row(self): return self.__row == 1

    def is_col(self): return self.__col == 1

    def adjoint(self, symbol:Optional[str]=None):
        if not self.is_square(): raise ValueError("adjoint is defined only for square matrices")
        cofactors = [[self.cofactor(i,j) for j in range(self.__col)] for i in range(self.__row)]
        return Matrix(cofactors, dtype=self.__dtype, symbol=symbol).T

    def trace(self):
        if not self.is_square(): raise ValueError("given matrix must be a square matrix number of rows and columns must be the same")
        total = 0
        for x in range(self.__row):
            for y in range(self.__col):
                if x == y: total += self.__matrix[x][y]
        return total

    def transpose(self): return self.T

    def flatten(self, symbol:Optional[str]=None):
        flattened = [item for row in self.__matrix for item in row]
        return Matrix([flattened], dtype=self.__dtype, symbol=symbol)

    def narrow(self, axis:int, start:int, length:int, symbol:Optional[str]=None):
        if axis not in [0,1]: raise ValueError(f"Invalid axis: {axis}. Expected '0' (rows) or '1' (columns).")
        if axis == 0:
            if start < 0: start = self.__row + start
            if start < 0 or start + length > self.__row: raise ValueError("Invalid start or length for row narrowing")
            narrowed_matrix = self.__matrix[start:start + length]
        elif axis == 1:
            if start < 0: self.__col + start
            if start < 0 or start + length > self.__col: raise ValueError("Invalid start or length for column narrowing")
            narrowed_matrix = [row[start:start + length] for row in self.__matrix]
        return Matrix(narrowed_matrix, dtype=self.__dtype, symbol=symbol)

    def reshape(self, rows:int, cols:int):
        if self.__size != rows * cols: raise ValueError(f"cannot reshape array of size {self.__size} into shape {(rows,cols)}")
        flattend = self.squeeze()
        self.__matrix = [flattend[i * cols:(i + 1) * cols] for i in range(rows)]
        self.__row = rows
        self.__col = cols
        self.__shape = (self.__row,self.__col)

    def view(self, shape:Tuple[int,int], symbol:Optional[str]=None):
        rows,cols = shape
        if self.__size != rows * cols: raise ValueError(f"cannot reshape array of size {self.__size} into shape {shape}")
        flattend = self.squeeze()
        return Matrix([flattend[i * cols:(i + 1) * cols] for i in range(rows)], dtype=self.__dtype, symbol=symbol)

    def lu_decomposition(self):
        if not self.is_square(): raise ValueError("LU decomposition is only defined for square matrices")
        n = self.__row
        L = zeros((n,n))
        U = zeros((n,n))
        for i in range(n):
            L[i,i] = 1
            for j in range(n): U[i,j] = self.__matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(n): L[j,i] = (self.__matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i,i]
        return L,U

    def sum(self, keepdim=False):
        if keepdim:
            new_mat = []
            for row in self.__matrix: new_mat.append([sum(row)])
            return Matrix(new_mat, dtype=self.__dtype)
        elif not keepdim:
            total = 0
            for row in self.__matrix:
                for x in row: total += x
            return total
        else: raise ValueError("invalid argument")

    def frobenius_norm(self):
        sum_of_sqrs = sum(x**2 for row in self.__matrix for x in row)
        return np.sqrt(sum_of_sqrs)

    def one_norm(self):
        col_sums = [sum(abs(x) for x in column) for column in zip(*self.__matrix)]
        return max(col_sums)

    def inf_norm(self):
        row_sums = [sum(abs(x) for x in row) for row in self.__matrix]
        return max(row_sums)

    def cholesky(self, symbol:Optional[str]=None):
        if self.__row != self.__col: raise ValueError("Matrix must be square for Cholesky decomposition")
        dense = np.array(self.__matrix)
        if not np.allclose(dense,dense.T): raise ValueError("Matrix must be symmetric for Cholesky decomposition")
        L = np.zeros_like(dense)
        for i in range(self.__row):
            for j in range(i + 1):
                sum_k = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j: L[i][j] = np.sqrt(dense[i][i] - sum_k)
                else: L[i][j] = (dense[i][j] - sum_k) / L[j][j]
        return Matrix(L.tolist(), dtype=self.__dtype, symbol=symbol)

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
                    for k in range(i,self.__col): mat[j][k] -= factor * mat[i][k]
        return rank

    def heatmap(self, title:str="Matrix HeatMap", cmap:str="viridis"):
        plt.figure(figsize = (8,6))
        sns.heatmap(self.__matrix,annot = True,fmt = "d",cmap = cmap,cbar = True)
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def graph(self, title:str="Graph Visualization of Matrix"):
        G = netx.Graph()
        for row in range(self.__row):
            for x in range(self.__col):
                if self.__matrix[row][x] != 0: G.add_edge(row,x,weight = self.__matrix[row][x])
        pos = netx.spring_layout(G)
        edge_labels = netx.get_edge_attributes(G,"weight")
        plt.figure(figsize = (8,6))
        netx.draw(G,pos,with_labels = True,node_color = "orange",edge_color = "blue",node_size = 500,alpha = 0.8)
        netx.draw_networkx_edge_labels(G,pos,edge_labels = edge_labels,font_color = "red")
        plt.title(title)
        plt.show()
