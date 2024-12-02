#!/usr/bin/python3
#
# Requirement:
# Python module RRDBNet_arch.py

# Import the Python modules.
import os
import torch
import RRDBNet_arch as arch

# Reset the terminal window.
os.system('reset')

# Set the torch print options for pretty printing.
torch.set_printoptions(threshold=0, linewidth=80, edgeitems=0, sci_mode=False)

# Define the upscaler model
model_file = "RRDB_ESRGAN_x4.pth"

# Load the upscaler model.
test_net = torch.load(model_file, weights_only=True)

# Print content and type.
print(test_net, "\n")
print(type(test_net), "\n")

# Print formatted content of the test net.
for k, v in test_net.items():
    print(("%-31s  ->  %s") % (str(k), str(test_net[k])))
