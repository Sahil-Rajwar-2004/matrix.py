#!/usr/bin/bash

pkg="matrix"

pip show "$pkg" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "$pkg lib is already installed"
    read -p "do you want to uninstall it and reinstall? (y/n): " choice
    if [ "$choice" != "y" ]; then
        echo "exiting without reinstalling!"
        exit 0
    fi
    echo "Uninstalling ($pkg)....."
    pip uninstall "$pkg" -y
else
    echo "$pkg lib isn't installed yet"
fi

echo "Initiating the installation process..."

if [ -e "./setup.py" ]; then
    echo "./setup.py has been found!"
else
    echo "setup.py file not found!"
    exit 1
fi

set -e

python3 ./setup.py sdist bdist_wheel
cd ./dist
pip install *.whl
cd ..
rm -rf dist build matrix.egg-info

pip show "$pkg" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "$pkg lib has been installed successfully"
else
    echo "something bad has happened! try to install manually"
    exit 1
fi
