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

# Create a ESRGAN model.
esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)

# Set the module in evaluation mode (not training mode).
esrgan_model.eval()

# Load the upscaler model.
test_net = torch.load(model_file, weights_only=True)

# Get the state dict.
test_state_dict = esrgan_model.load_state_dict(torch.load(model_file, weights_only=True))

# Print content and type.
print(test_state_dict, "\n")
print(type(test_state_dict), "\n")
