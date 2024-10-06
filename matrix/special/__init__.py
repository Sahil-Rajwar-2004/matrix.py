from typing import Union,Optional
import numpy as np
import math
import matrix

Matrix = matrix.Matrix

def __res_dtype__(matrix:Matrix, other:Matrix):
    if not isinstance(other, Matrix):
        if matrix.dtype == "complex" or type(other) == complex: return complex
        elif matrix.dtype == "bool" and type(other) == bool: return bool
        elif (matrix.dtype == "int" and type(other) == float) or (matrix.dtype == "float" and type(other) == int): return float
        elif (matrix.dtype == "bool" and type(other) == int) or (matrix.dtype == "int" and type(other) == bool): return int
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

def exp2(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(2 ** matrix[row,col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def exp3(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(3 ** matrix[row,col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def expn(matrix:Matrix, n:Union[Matrix,int,float,complex], symbol:Optional[str]=None):
    if isinstance(n,Matrix):
        if not (matrix.shape == n.shape): raise ValueError(f"the shape of the two matrices must be the same, {matrix.shape} != {n.shape}")
        new_mat = []
        for row in range(matrix.row):
            buffer = []
            for col in range(matrix.col): buffer.append(n[row,col] ** matrix[row,col])
            new_mat.append(buffer)
        return Matrix(new_mat, dtype=__res_dtype__(matrix,n), symbol=symbol)
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(n ** matrix[row,col])
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def expit(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append((1 + np.exp(-matrix[row,col]))**-1)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def expm1(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col): buffer.append(np.exp(matrix[row,col] - 1))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def entr(matrix:Matrix, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row,col]
            if value > 0: buffer.append(-value * np.log(value))
            elif value == 0: buffer.append(0)
            elif value < 0: buffer.append(-float("inf"))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def i0(matrix:Matrix, terms:int=50, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row,col]
            result = 0
            for k in range(terms):
                term = ((value ** 2) / 4) ** k / math.factorial(k) ** 2
                result += term
            buffer.append(result)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def i0e(matrix:Matrix, terms:int=50, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row,col]
            result = 0
            for k in range(terms):
                term = ((value ** 2) / 4) ** k / math.factorial(k) ** 2
                result += term
            buffer.append(result * np.exp(-value))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)   

def i1(matrix:Matrix, terms:int=50, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row,col]
            result = 0
            for k in range(terms):
                term = ((value ** 2) / 4) ** k / (math.factorial(k) * math.factorial(k + 1))
                result += term
            buffer.append(result * value / 2)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)

def i1e(matrix:Matrix, terms:int=50, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row,col]
            result = 0
            for k in range(terms):
                term = ((value ** 2) / 4) ** k / (math.factorial(k) * math.factorial(k + 1))
                result += term
            buffer.append(result * value * np.exp(-value) / 2)
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)
   
def logit(matrix:Matrix, eps=None, symbol:Optional[str]=None):
    new_mat = []
    for row in range(matrix.row):
        buffer = []
        for col in range(matrix.col):
            value = matrix[row, col]
            if eps is not None: value = np.clip(value, eps, 1 - eps)
            buffer.append(np.log(value / (1 - value)))
        new_mat.append(buffer)
    return Matrix(new_mat, dtype=__res_dtype_self__(matrix), symbol=symbol)
