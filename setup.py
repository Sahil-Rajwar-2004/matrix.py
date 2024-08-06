from setuptools import setup,find_packages


setup(
    name = "matrix",
    version = "0.5.3",
    author = "Sahil Rajwar",
    description = "python library for doing calculations with matrices",
    long_description = open("README.md","r",encoding = "utf-8").read(),
    long_description_content_type = "text/markdown",
    license = "MIT",
    url = "https://github.com/Sahil-Rajwar-2004/matrix.py",
    packages = find_packages(),
    install_requires = ["numpy","matplotlib","seaborn","networkx"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)

