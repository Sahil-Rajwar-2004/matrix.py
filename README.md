# matrix.py

## ***python library for calculations with matrices***

## Installation
```bash
pip install <latest-matrix-version-wheel-file>
```

## Usage
```python
from matrix import Matrix

x = Matrix([[1,2,3],[4,5,6]])
y = Matrix([[2,3,4],[5,6,7]])

print(x)
print(x.numpy())
print((x + y).numpy())

```

## OutPut
```bash
<'Matrix' object at 0x7fdfc2d87fa0 size=6 shape=(2, 3)>
array([[1, 2, 3],
       [4, 5, 6]])
[[ 3  5  7]
 [ 9 11 13]]
```

