from setuptools import setup, find_packages

setup(
    name = "matrix",
    version = "0.1.2",
    author = "Sahil Rajwar",
    description = "python library for doing calculations with matrices",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    license = "MIT",
    url = "https://github.com/Sahil-Rajwar-2004/matrix.py",
    packages = find_packages(),
    requires = ["numpy"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)

