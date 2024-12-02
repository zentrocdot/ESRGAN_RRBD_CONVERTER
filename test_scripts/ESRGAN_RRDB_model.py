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
torch.set_printoptions(threshold=0, linewidth=80, edgeitems=0, sci_mode=False, precision=4)

# Create a ESRGAN model.
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)

# Set the module in evaluation mode (not training mode).
esrgan_model.eval()

# Create a test net on base of the ESRGAN model.
test_net = esrgan_model.state_dict()

# Print content and type.
print(test_net, "\n")
print(type(test_net), "\n")

# Print formatted content of the test net.
for k, v in test_net.items():
    print(("%-31s  ->  %s") % (str(k), str(test_net[k])))
