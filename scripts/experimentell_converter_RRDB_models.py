#!/usr/bin/python3
#
# Version 0.0.0.1
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
import traceback

# Import the third party Python module.
import torch

# Import Python module from directory.
import RRDBNet_arch as arch

# Supress the future warnings.
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Get the model name from the command line.
model_name = sys.argv[1]

# Get basename and extension.
file_name = os.path.basename(model_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = "RRDB_" + basename + extension

# Prepare model data.
model_content = torch.load(file_name)
# If some keywords not in the model content it must be a new model.
weight_str = 'model.0.weight'
bias_str = 'model.0.bias'
# Check the keywords in the model content.
if not (weight_str and bias_str) in model_content:
    print("Nothing to do! Is yet a NEW ESRGAN model. Bye!")
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
print("Reference keywords to found keywords:")
print(ref_list)
print(conv_list)
# Create the conversion table dict.
CRT_DICT = {
            "conv_first.weight": "model.0.weight",
            "conv_first.bias": "model.0.bias",
            "trunk_conv.weight": "model.1.sub.23.weight",
            "trunk_conv.bias": "model.1.sub.23.bias",
            "upconv1.weight": "model.3.weight",
            "upconv1.bias": "model.3.bias",
            "upconv2.weight": "model.6.weight",
            "upconv2.bias": "model.6.bias",
            #"HRconv.weight": "model.8.weight",
            #"HRconv.bias": "model.8.bias",
            #"conv_last.weight": "model.10.weight",
            #"conv_last.bias": "model.10.bias"
            "HRconv.weight": conv_list[0],
            "HRconv.bias": conv_list[1],
            "conv_last.weight": conv_list[2],
            "conv_last.bias": conv_list[3]
           }

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(file_name, save_name, pretrained_net):
    '''Main script function.'''
    # Print input and output name.
    print("Input:", file_name)
    print("Output:", save_name)
    # Start conversion.
    print("Start conversion ...")
    # Set pretrained net name.
    pretrained_net = torch.load(file_name)
    # Start model conversion.
    crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    crt_net = crt_model.state_dict()
    # Create a new cleaned up dictionary.
    load_net_clean = {}
    for k, v in pretrained_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    pretrained_net = load_net_clean
    # Create a list with keys from crt_net.
    tbd = []
    for k, v in crt_net.items():
        tbd.append(k)
    # Loop over key/value pairs from crt_net.
    for k, v in crt_net.items():
        if k in pretrained_net and pretrained_net[k].size() == v.size():
            crt_net[k] = pretrained_net[k]
            tbd.remove(k)
    # Set first conv layer key/values.
    #crt_net['conv_first.weight'] = pretrained_net['model.0.weight']
    #crt_net['conv_first.bias'] = pretrained_net['model.0.bias']
    # Loop over the copied tbd list.
    for k in tbd.copy():
        if 'RDB' in k:
            ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
            if '.weight' in k:
                ori_k = ori_k.replace('.weight', '.0.weight')
            elif '.bias' in k:
                ori_k = ori_k.replace('.bias', '.0.bias')
            crt_net[k] = pretrained_net[ori_k]
            tbd.remove(k)
    # Loop over key/value pairs.
    for key, value in CRT_DICT.items():
        try:
            crt_net[key] = pretrained_net[value]
        except Exception as err:
            #print("ERROR:", err)
            print(traceback.format_exc())
    # Save new model.
    torch.save(crt_net, save_name)
    # print message.
    print("... conversion completed!")

# Execute the script as module or as programme.
if __name__ == "__main__":
    # Call the main function.
    main(file_name, save_name, model_content)
