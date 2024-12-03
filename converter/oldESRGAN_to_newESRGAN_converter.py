#!/usr/bin/python3
'''Experimental ESRGAN converter.'''
# pylint: disable=invalid-name
# pylint: disable=unneeded-not
# pylint: disable=unused-variable
# pylint: disable=broad-except
# pylint: disable=redefined-outer-name
#
# Version 0.0.0.3
#
# (C) 2024 zentrocdot
#
# Description:
# ------------
#
# First two keys in an old ESRGAN model are:
#
#  Old model          |    New model
# "model.0.weight"   -->  "conv_first.weight"
# "model.0.bias"     -->  "conv_first.bias"
#
# Last four keys in an old ESRGAN model are:
#
#  Old model          |    New model
# "model.8.weight"   -->  "HRconv.weight"
# "model.8.bias"     -->  "HRconv.bias"
# "model.10.weight"  -->  "conv_last.weight"
# "model.10.bias"    -->  "conv_last.bias"
#
# Sometimes last four lines look like:
#
#  Old model          |    New model
# "model.11.weight"  -->  "HRconv.weight"
# "model.11.bias"    -->  "HRconv.bias"
# "model.13.weight"  -->  "conv_last.weight"
# "model.13.bias"    -->  "conv_last.bias"
#
# Approach for handling this behaviour:
# We read all keys from the given model.
# Then we exchange the last four keywords.

# Import the standard Python modules.
import os
import sys
import warnings
import traceback
from collections import OrderedDict

# Import the third party Python module.
import torch

# Supress the future warnings.
warnings.filterwarnings('ignore', category=FutureWarning)

# Get the model name from the command line.
model_name = sys.argv[1]

if model_name is None:
    print("No model given. Bye!")
    os._exit(1)

# Get basename and extension.
file_name = os.path.basename(model_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = basename + "_CVTD" + extension

# Print header to screen.
print("***  ESRGAN CONVERTER  ***")

# Prepare model data.
model_content = torch.load(file_name)

# If some keywords not in the model content it must be a new model.
weight_str = 'model.0.weight'
bias_str = 'model.0.bias'
# Check the keywords in the model content.
if not (weight_str and bias_str) in model_content:
    print("Nothing to do! Is a NEW ESRGAN, a RealESRGAN or an unknown model. Bye!")
    os._exit(127)

# Loop over the keys in model content.
key_list = []
for key in model_content:
    key_list.append(key)
# Initialise N.
N = 4
# Using list slicing.
ref_list = ["model.8.weight", "model.8.bias",
            "model.10.weight","model.10.bias"]
conv_list = key_list[-N:]
res = ref_list == conv_list
if res == False:
    print("Reference keywords to found keywords:")
    print(ref_list)
    print(conv_list)
# Create the conversion table dict.
# Parametrised variables:
#   "HRconv.weight": "model.8.weight",
#   "HRconv.bias": "model.8.bias",
#   "conv_last.weight": "model.10.weight",
#   "conv_last.bias": "model.10.bias"
CKT_DICT = {
            "model.0.weight": "conv_first.weight",
            "model.0.bias": "conv_first.bias",
            "model.1.sub.23.weight": "trunk_conv.weight",
            "model.1.sub.23.bias": "trunk_conv.bias",
            "model.3.weight": "upconv1.weight",
            "model.3.bias": "upconv1.bias",
            "model.6.weight": "upconv2.weight",
            "model.6.bias": "upconv2.bias",
            conv_list[0]: "HRconv.weight",
            conv_list[1]: "HRconv.bias",
            conv_list[2]: "conv_last.weight",
            conv_list[3]: "conv_last.bias"
           }

# Set error message string.
ERR_STR_0 = "Could not finish converting the model. Bye!"
ERR_STR_1 = "Could not saving the converted model. Bye!"

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(file_name: str, save_name: str, pretrained_net: OrderedDict) -> None:
    '''Main script function.'''
    # Set the exchange patterns and string.
    pattern = ['model.', '.RDB', '.weight', '.bias']
    substr = ['model.1.sub.', 'RRDB_trunk.',
              '.0.weight', '.weight',
              '.0.bias', '.bias']
    # Initilise the local variable.
    new_k = None
    # Print input and output name.
    print("Input:", file_name)
    print("Output:", save_name)
    # Start conversion.
    print("Start conversion ...")
    # Load pretrained net.
    pretrained_net = torch.load(file_name)
    # Create a new ordered dictionary.
    new_net_clean = OrderedDict()
    # Loop over the items of pretrained_net.
    for k, v in pretrained_net.items():
        # Check if k starts with substring model.
        if k.startswith(pattern[0]):
            # Check if in key is the pattern RDB.
            if pattern[1] in k:
                new_k = (k.replace(substr[0], substr[1]))
                # Rewrite key containing weight and bias.
                if pattern[2] in new_k:
                    new_k = new_k.replace(substr[2], substr[3])
                elif pattern[3] in k:
                    new_k = new_k.replace(substr[4], substr[5])
            else:
                try:
                    new_k = CKT_DICT[k]
                except KeyError as err:
                    #print(traceback.format_exc())
                    print("KeyError:", err)
                    print(ERR_STR_0)
                    return None
        # Try to assign tensor to key.
            new_net_clean[new_k] = v
    # Overwrite the pretrained net.
    pretrained_net = new_net_clean
    # Try to save new model.
    try:
        torch.save(pretrained_net, save_name)
    except RuntimeError as err:
        #print("Error:", err)
        #print(traceback.format_exc())
        print(ERR_STR_1)
        return None
    # Print message.
    print("... conversion completed!")
    # Return None.
    return None

# Execute the script as module or as programme.
if __name__ == "__main__":
    # Call the main function.
    main(file_name, save_name, model_content)
