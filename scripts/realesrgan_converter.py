#!/usr/bin/python
'''ESRGan to RealESRGAN Converter.'''
#
# ESRGAN to RealESRGAN Converter
# Version 0.0.0.1
#
# pylint: disable=useless-return
# pylint: disable=invalid-name
# pylint: disable=bare-except
##pylint:#disable=redefined-outer-name
# pylint: disable=unused-import

# Import the standard Python modules.
import os
import sys
import warnings
import traceback
from collections import OrderedDict

# Import the third party Python module.
import torch

# Import the Python module from directory.
import RRDBNet_arch as arch

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

# ------------
# Reset screen
# ------------
def reset_term() -> None:
    '''Reset the terminal window.'''
    ESC_SEQ = "\33c"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# ------------
# Clear screen
# ------------
def clear_term() -> None:
    '''Clear the terminal window.'''
    ESC_SEQ = "\33[2J\33[H"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# Define the conversion table.
CRT_DICT = {
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

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(filename, savename):
    '''Main script function'''
    # Print the input and the output name.
    print("Input File:", filename)
    print("Output File:", savename)
    # Try to get pretrained model from the model file.
    # Type: pretrained net -> collections.OrderedDict.
    try:
        pretrained_net = torch.load(file_name)
    except:
        print(pretrained_net)
        print("Conversion not possible. Bye!")
        return None
    # Print a message to the screen.
    print("Start conversion ...")
    # Create a new model.
    #crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    #crt_net = crt_model.state_dict()
    # Create a new cleaned up dictionary.
    #load_net_clean = {}
    load_net_clean = OrderedDict()
    for k, v in pretrained_net.items():
        if k.startswith('RRDB_trunk.'):
            ori_k = (k.replace('RRDB_trunk.', 'body.')).lower()
            load_net_clean[ori_k] = v
        else:
            ori_k = CRT_DICT[k]
            load_net_clean[ori_k] = v
    pretrained_net = load_net_clean
    print(type(pretrained_net))
    # Create a new dictionary.
    pretrained_net = {"params_ema": pretrained_net}
    # Save the new model.
    torch.save(pretrained_net, savename)
    # Print a message to the screen.
    print("... conversion completed!")
    # Return None
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(file_name, save_name)
