#!/usr/bin/python
''' Structured reading and writing data from .pt file.'''
#
# Version 0.0.0.1
#
# pylint: disable=wrong-import-position
# pylint: disable=consider-using-sys-exit

# Import module.
import warnings

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import module.
import torch

# Set the file.
PTH_FILE = "filename.pth"

# Main script function.
def main(filename):
    '''Main script function'''
    # Load pt file into data.
    data = torch.load(filename)
    # Loop over key and value in dict.
    for param in data:
        # print key and value.
        print(param, data[param])

# Execute as module as well as a program.
if __name__ == "__main__":
    # Call main function.
    main(PTH_FILE)
