#!/bin/bash

# Create a virtual environment
python -m venv fsgc
source ./fsgc/bin/activate

# Check if a nVidia GPU is present on the system
result=`nvidia-smi -a | grep --color -E GTX`
available=$(test $? != 0 && echo "0" || echo "")

# Install all dependencies
if [ -z ${a} ]; then
	pip install torch==1.12.1
else
	pip install torch==
fi;