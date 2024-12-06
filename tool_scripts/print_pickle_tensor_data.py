#!/usr/bin/python
''' Print structure of .pt/.pth files.'''
#
# Print Pickle Tensor File Structure
# Version 0.0.0.1
#
# (C) 2024, zentrocdot
#
# This script is licensed under the terms of the MIT License:
# https://github.com/zentrocdot/ESRGAN_RRBD_CONVERTER#MIT-1-ov-file
#
# Description:
# Check if a file is given file via the command line argument. Then it
# it is checked if the file extension is a valid extension. Afterwards
# the script tries to load the Pickle Tensor files and tries to print
# the loaded data into the terminal window.
#
# pylint: disable=protected-access
# pylint: disable=broad-except
# pylint: disable=unused-variable
# pylint: disable=invalid-name

# Import the standard Python modules.
import os
import sys
import warnings
import traceback

# Import the third party Python module.
import torch

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set torch print options.
# Default: threshold=10_000, linewidth=80, precision=4, edgeitems=3
torch.set_printoptions(threshold=10, precision=4, edgeitems=2, linewidth=160)

# Get the Pickle Tensor file name from the command line.
if len(sys.argv) > 1:
    PT_FILE = sys.argv[1]
else:
    exit_message = "No file given. Bye!"
    print(exit_message)
    os._exit(1)

# Check the file extension.
if not PT_FILE.lower().endswith(('.pt', '.pth')):
    exit_message = "No valid file extension found. Bye!"
    print(exit_message)
    os._exit(2)

# ------------------------------------------
# Function pt_file_type()
# Magic Numbers Zip Files:
# PK\x03\x04, PK\x05\x06 (empty), PK\x07\x08
# PK = \x50\x4B
# ------------------------------------------
def pt_file_type(filename):
    '''Get the file type of a Pickle Tensor file.'''
    # Initialise the file type variable.
    file_type = "unknown"
    with open(filename, 'rb') as file:
        content = file.read(2)
        if content.find(b'\x80\x02') != -1:
            file_type = "binary"
        elif content.find(b'\x50\x4B') != -1:
            file_type = "zip"
    # Return the file type by name.
    return file_type

# Main script function.
def main(filename: str) -> None:
    '''Main script function'''
    # Try to load a Pickle Tensor file into data.
    try:
        data = torch.load(filename)
    except Exception as err:
        print("*** Begin of traceback output")
        print(traceback.format_exc())
        sys.stdout.write("\033[A")
        sys.stdout.flush()
        print("*** End of traceback output")
        return None
    # Loop over key and value in dict.
    for param in data:
        # print key and value.
        print(param, data[param])
    # Print Pickle Tensor file type.
    print("Type of Pickle Tensor file:", pt_file_type(filename))
    # Return None.
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Call main function.
    main(PT_FILE)
