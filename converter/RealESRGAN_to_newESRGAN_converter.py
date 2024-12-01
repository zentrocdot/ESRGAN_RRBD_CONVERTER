#!/usr/bin/python
'''RealESRGAN to ESRGAN Converter.'''
#
# RealESRGAN to ESRGAN Converter
# Version 0.0.0.1
#
# pylint: disable=useless-return
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

# Import standard Python modules.
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
save_name = basename + "_CONV_NEW" + extension

# Reset screen.
def reset_term() -> None:
    '''Reset the terminal window.'''
    ESC_SEQ = "\33c"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# Clear screen.
def clear_term() -> None:
    '''Clear the terminal window.'''
    ESC_SEQ = "\33[2J\33[H"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# Create a global weight/bias key/value dictionary.
CKT_DICT = {
            "conv_first.weight": "conv_first.weight",
            "conv_first.bias": "conv_first.bias",
            "conv_body.weight": "trunk_conv.weight",
            "conv_body.bias": "trunk_conv.bias",
            "conv_up1.weight": "upconv1.weight",
            "conv_up1.bias": "upconv1.bias",
            "conv_up2.weight": "upconv2.weight",
            "conv_up2.bias": "upconv2.bias",
            "conv_hr.weight": "HRconv.weight",
            "conv_hr.bias": "HRconv.bias",
            "conv_last.weight": "conv_last.weight",
            "conv_last.bias": "conv_last.bias"
           }

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(file_name, save_name):
    '''Main script function'''
    # Print input and output name.
    print("Input File:", file_name)
    print("Output File:", save_name)
    # Load the model data (Type collections.OrderedDict).
    model_data = torch.load(file_name)
    # Try to get pretrained model from the model data.
    print(list(model_data.keys())[0])
    key_word = list(model_data.keys())[0]
    try:
        pretrained_net = model_data[key_word]
    except:
        print(model_data)
        print("Conversion not possible. Bye!")
        return None
    # Start conversion.
    print("Start conversion ...")
    # Create a new ordered dictionary.
    new_net_clean = OrderedDict()
    # Loop over the items of pretrained_net.
    for k, v in pretrained_net.items():
        if k.startswith('body.'):
            #print(k)
            old_sub_str = 'body.'
            new_sub_str = 'RRDB_trunk.'
            new_k = (k.replace(old_sub_str, new_sub_str))
            new_k = (new_k.replace('.rdb', '.RDB'))
            new_net_clean[new_k] = v
            print(new_k)
        else:
            new_k = CKT_DICT[k]
            print(new_k)
            new_net_clean[new_k] = v
    # Overwrite pretrained net.
    pretrained_net = new_net_clean
    # Save new model.
    torch.save(pretrained_net, save_name)
    # print message.
    print("... conversion completed!")
    # Return None
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(file_name, save_name)
