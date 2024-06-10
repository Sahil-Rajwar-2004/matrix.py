# matrix.py

## ***python library for calculations with matrices***

## Installation
click here to download latest version :: [v-0.4.2](https://github.com/Sahil-Rajwar-2004/matrix.py/releases/download/v0.4.2/matrix-0.4.2-py3-none-any.whl)
```bash
pip install <latest-version-whl>
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

