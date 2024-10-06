# matrix.py

## ***A Python library for performing advanced matrix calculations***

## Implementation
```python
from matrix import Matrix

x = Matrix([[1,2,3],[4,5,6]], dtype=int)                     # default symbol = None
y = Matrix([[2,3,4],[5,6,7]], symbol="A")                    # default dtype = float

print(x())                                                   # <'Matrix' object at 0x7fdfc2d87fa0 dtype=int size=6 shape=(2, 3) symbol=None>
print(y())                                                   # <'Matrix' object at 0x7fb588637f10 dtype=float size=6 shape=(2, 3) symbol=A>
print(x.numpy())                                             # array([[1, 2, 3], [4, 5, 6]])
print((x + y)())                                             # <'Matrix' object at 0x7fb544042140 dtype=float size=6 shape=(2, 3) symbol=None>
print(x + y)

# Output
"""
Matrix([
    [ 3.00000,  5.00000,  7.00000],
    [ 9.00000, 11.00000, 13.00000]
], dtype=float, symbol=None, shape=(2, 3))
"""

```


## Installation Instruction

1. clone this repo  
```bash
git clone "https://github.com/Sahil-Rajwar-2004/matrix.py.git"
```

2. navigate to matrix directory that you just downloaded/cloned

3. see below instruction for different OS

## Windows Users

4. If you are on Window OS then run the `install.ps1` script  
```powershell
pwsh install.ps1
```

## OR Linux / MacOS Users

4. If you are on Linux / MacOS then run the `install.sh` script  
```bash
bash install.sh
```


## Contributing
I am currently the sole developer and maintainer of this project. While contributions are welcome, please reach out before starting any major work to ensure that your ideas align with the project's goals. You can open a discussion or contact me directly through the repository.


## Reporting Issues
If you encounter any issues or bugs while using the library, please report them by opening an issue on the [GitHub repository](https://github.com/Sahil-Rajwar-2004/matrix.py/issues). Include a detailed description of the problem, steps to reproduce it, and any relevant error messages or screenshots.


## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Sahil-Rajwar-2004/matrix.py/blob/master/LICENSE) file for more information.
