# matrix.py

## ***python library for calculations with matrices***

## Usage
```python
from matrix import Matrix

x = Matrix([[1,2,3],[4,5,6]])
y = Matrix([[2,3,4],[5,6,7]])

print(x)                    # <'Matrix' object at 0x7fdfc2d87fa0 size=6 shape=(2, 3)>
print(x.numpy())            # array([[1, 2, 3], [4, 5, 6]])
print((x + y).numpy())      # [[ 3 5 7] [ 9 11 13]]
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
