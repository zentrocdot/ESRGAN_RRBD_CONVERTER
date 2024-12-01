#!/usr/bin/python3
'''New ESRGAN to RealESRGAN Converter.

ESRGAN is available in form of Pickle Tensor files in the old and new
architecture. Shortly spoken OLD ESRGAN or NEW ESRGAN. Both formats are
incompatible with each other. RealESRGAN is a further development of
ESRGAN. This format is incompatible with the aforementioned format. This
script converts NEW ESRGAN to RealESRGAN.

Weight, Bias and other data in RealESRGAN are given in an OrderedDict as
value to a key in a dict. This is the main difference to ESRGAN. In ESRGAN
Weight, Bias and other data are given as OrderedDict. The main goal of the
converter is the translation of the different key words to the Tensors as
values.

'''
# ESRGAN to RealESRGAN Converter
# Version 0.0.0.1
#
# (C) 2024, zentrocdot
#
# This script is licensed under the terms of the MIT License:
# https://github.com/zentrocdot/ESRGAN_RRBD_CONVERTER#MIT-1-ov-file
#
# pylint: disable=useless-return
# pylint: disable=invalid-name
# pylint: disable=bare-except
# pylint: disable=unused-import

# Import the standard Python modules.
import os
import sys
import warnings
import traceback
from collections import OrderedDict

# Import the third party Python module.
import torch

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the model name from the command line.
model_file_name = sys.argv[1]

# Get basename and extension.
file_name = os.path.basename(model_file_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = basename + "_REAL" + extension

# Define the conversion key table in form of a dict.
CKT_DICT = {
            "conv_first.weight": "conv_first.weight",
            "conv_first.bias": "conv_first.bias",
            "trunk_conv.weight": "conv_body.weight",
            "trunk_conv.bias": "conv_body.bias",
            "upconv1.weight": "conv_up1.weight",
            "upconv1.bias": "conv_up1.bias",
            "upconv2.weight": "conv_up2.weight",
            "upconv2.bias": "conv_up2.bias",
            "HRconv.weight": "conv_hr.weight",
            "HRconv.bias": "conv_hr.bias",
            "conv_last.weight": "conv_last.weight",
            "conv_last.bias": "conv_last.bias"
           }

# ------------------------------
# Helper function. Reset screen.
# ------------------------------
def reset_term() -> None:
    '''Reset the terminal window.'''
    ESC_SEQ = "\33c"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# -----------------------------
# Helper function Clear screen.
# -----------------------------
def clear_term() -> None:
    '''Clear the terminal window.'''
    ESC_SEQ = "\33[2J\33[H"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(filename: str, savename: str) -> None:
    '''Main script function'''
    # Print the input and the output name.
    print("Input File:", filename)
    print("Output File:", savename)
    # Try to get pretrained model from the model file.
    # Type: pretrained net -> collections.OrderedDict.
    try:
        pretrained_net = torch.load(file_name)
    except:
        # Print file content for review.
        print(pretrained_net)
        # Print farewell message.
        print("Conversion not possible. Bye!")
        # Return None
        return None
    # Print a message to the screen.
    print("Start conversion ...")
    # Create a new ordered dictionary.
    new_net_clean = OrderedDict()
    # Loop over the items of pretrained_net.
    for k, v in pretrained_net.items():
        old_sub_str = 'RRDB_trunk.'
        new_sub_str = 'body.'
        # Replace substrings on occurence.
        if k.startswith(old_sub_str):
            new_k = (k.replace(old_sub_str, new_sub_str)).lower()
            new_net_clean[new_k] = v
        # Use the conversion key table.
        else:
            new_k = CKT_DICT[k]
            new_net_clean[new_k] = v
    # Overwrite pretrained_net with load_net_clean.
    pretrained_net = new_net_clean
    # Create a new dictionary. This is the dictionary introducer.
    key_word = "params_ema"
    pretrained_net = {key_word: pretrained_net}
    # Save the new model.
    torch.save(pretrained_net, savename)
    # Print a message to the screen.
    print("... conversion completed!")
    # Return None
    return None

# Execute as module as well as a programme.
if __name__ == "__main__":
    # Clear or reset terminal. Use what needed.
    #clear_term()
    reset_term()
    # Call the main function.
    main(file_name, save_name)
