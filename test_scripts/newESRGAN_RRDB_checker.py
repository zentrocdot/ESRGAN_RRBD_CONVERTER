#!/usr/bin/python3
#
# Requirement:
# Python module RRDBNet_arch.py

# Import the Python modules.
import os
import sys
import torch
import RRDBNet_arch as arch

# Reset the terminal window.
os.system('reset')

# Read the filename.
fn = sys.argv[1]

# Create a ESRGAN model.
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)

# Set the module in evaluation mode (not training mode).
esrgan_model.eval()

# Try to load the model.
try:
    test_net = torch.load(fn, weights_only=True)
except:
    print("Could not load the model! No valid model!")

# Try to get the state dict.
try:
    test_state_dict = esrgan_model.load_state_dict(test_net)
    # Print content and type.
    string = "<class 'torch.nn.modules.module._IncompatibleKeys'>"
    if str(type(test_state_dict)) == string:
        print("Valid new ESRGAN model!")
    #print(test_state_dict)
    #print(type(test_state_dict), "\n")
except RuntimeError as err:
    print("Could not extract the state dictionary from the model! No valid new ESRGAN model.")

