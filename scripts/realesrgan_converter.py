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

# Import the third party Python module.
import torch

# Import the Python module from directory.
import RRDBNet_arch as arch

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set torch print options.
#torch.set_printoptions(threshold=10_000)

# Get the model name from the command line.
model_file_name = sys.argv[1]

# Get basename and extension.
file_name = os.path.basename(model_file_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = "RRDB_" + basename + "_CONV" + extension

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
# Key name is unchanged for:
# "conv_first.weight": "conv_first.weight",
# "conv_first.bias": "conv_first.bias",
# "conv_last.weight": "conv_last.weight",
# "conv_last.bias": "conv_last.bias"
CRT_DICT = {
            "trunk_conv.weight": "conv_body.weight",
            "trunk_conv.bias": "conv_body.bias",
            "upconv1.weight": "conv_up1.weight",
            "upconv1.bias": "conv_up1.bias",
            "upconv2.weight": "conv_up2.weight",
            "upconv2.bias": "conv_up2.bias",
            "HRconv.weight": "conv_hr.weight",
            "HRconv.bias": "conv_hr.bias",
           }

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(file_name, save_name):
    '''Main script function'''
    # Set the key word.
    #key_word = "params_ema"
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
    # Create a model.
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
    # Loop over the copied tbd list.
    for k in tbd.copy():
        if 'RDB' in k:
            #print(k)
            ori_k = (k.replace('RRDB_trunk.', 'body.')).lower()
            #print(ori_k)
            try:
                crt_net[k] = pretrained_net[ori_k]
            except:
                pass
            #print(pretrained_net[ori_k])
            tbd.remove(k)
    # Try to set values to key.
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
    # Return None
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(file_name, save_name)
