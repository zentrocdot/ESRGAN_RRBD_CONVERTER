#!/usr/bin/python3
'''Analysis of the old ESRGAN data structure.'''
# pylint: disable=useless-return
# pylint: disable=bare-except
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=too-many-nested-blocks
#
# old ESRGAN Model Data Analysis
# Version 0.0.0.1
#
# To-Do:
# Sanitize and optimise script. Add a description.
# 

# Import standard Python modules.
import re
import os
import sys
import warnings

# Import third party module.
import torch

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set torch print options.
torch.set_printoptions(threshold=10_000)

# Get the model name from the command line.
# To-Do: check if it is a file (zip/binary).
# To-Do: check if it is a pt or pth file.
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    exit_message = "No model on the command line given. Bye!"
    print(exit_message)
    os._exit(1)

# Set the color strings.
COL_DEFAULT = "\33[49m"
COL_RED = "\33[41m"
COL_GREEN = "\33[42m"
COL_YELLOW = "\33[43m"
COL_BLUE = "\33[44m"
COL_MAGENTA = "\33[45m"
COL_CYAN = "\33[46m"
COL_GRAY = "\33[47m"
COL_DARK_GRAY = "\33[100m"

# Set some strings.
error_str = "\n\33[41mERROR: Check the data structure!\33[49m"
warn_str_0 = "\n\33[46mFound UNKNOWN (OLD) ESRGAN RRBD model. Check the data!\33[49m"
warn_str_1 = "\n\33[46mFound UNKNOWN (NEW) ESRGAN RRBD model. Check the data!\33[49m"
info_str_0 = "\n\33[45mFound old ESRGAN RRBD model\33[49m"
info_str_1 = "\n\33[45mFound new ESRGAN RRBD model\33[49m"

# Set some strings.
weight_str_0 = 'model.0.weight'
bias_str_0 = 'model.0.bias'
weight_str_1 = 'conv_first.weight'
bias_str_1 = 'conv_first.bias'

# Define the regular expressions.
body_regex = r"^model.1.sub.\b([0-9]|[1][3-9]|[12][0-2])\b.RDB[1-3].conv[0-5].0.(?:weight|bias)$"
head_foot_regex = r"^model.(1.sub.23|[01368]+[0]*).(?:weight|bias)$"

# Compile the regular expressions for use.
pattern_body = re.compile(body_regex)
pattern_head_foot = re.compile(head_foot_regex)

# Set key word dict.
key_word_dict = {"RRDB_trunk.0.": "NEW ESRGAN",
                 "model.1.sub.": "OLD ESRGAN",
                 "body.": "RealESRGAN"}
key_word_list = list(key_word_dict.keys())

# Set pre/post tensor dict and pre/post string list.
ppdict = {"model.0.weight": [64, 3, 3, 3],
          "model.0.bias": [64],
          "model.1.sub.23.weight": [64, 64, 3, 3],
          "model.1.sub.23.bias": [64],
          "model.3.weight": [64, 64, 3, 3],
          "model.3.bias": [64],
          "model.6.weight": [64, 64, 3, 3],
          "model.6.bias": [64],
          "model.8.weight": [64, 64, 3, 3],
          "model.8.bias": [64],
          "model.10.weight": [3, 64, 3, 3],
          "model.10.bias": [3]}
pplist = list(ppdict.keys())

# New approach.
conv_tens = {"conv1": {"weight": [32, 64, 3, 3], "bias": [32]},
             "conv2": {"weight": [32, 96, 3, 3], "bias": [32]},
             "conv3": {"weight": [32, 128, 3, 3], "bias": [32]},
             "conv4": {"weight": [32, 160, 3, 3], "bias": [32]},
             "conv5": {"weight": [64, 192, 3, 3], "bias": [64]}}

# Set model, weight and bias pattern.
PAT_W = "weight"
PAT_B = "bias"
PAT_MOD = "model"

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

# --------------------
# Function file_type()
# Magic Numbers Zip Files:
# PK\x03\x04
# PK\x05\x06 (empty)
# PK\x07\x08
# PK = \x50\x4B
# --------------------
def file_type(filename):
    file_type = "unknown"
    with open(filename, 'rb') as file:
        content = file.read(2)
        if content.find(b'\x80\x02') != -1:
            file_type = "binary"
        elif content.find(b'\x50\x4B') != -1:
            file_type = "zip"
    return file_type

# ---------------------
# Function check_keys()
# ---------------------
def check_keys(data):
    '''Check keys.'''
    ret_val = False
    ret_arr = []
    key_count = 0
    for key in data:
        ret_arr.append(key)
        key_count += 1
        if key.startswith(PAT_MOD):
            test = re.match(pattern_body, key)
            if test:
                ret_arr.remove(key)
            else:
                test = re.match(pattern_head_foot, key)
                if test:
                    ret_arr.remove(key)
    # On no mismatch set True.
    if not ret_arr:
        ret_val = True
    # Return True/False and mismatching data.
    return ret_val, ret_arr, key_count

# ------------------------
# Function check_tensors()
# ------------------------
def check_tensors(data):
    '''Check keys.'''
    # Initialise some variables.
    ret_val = False
    ret_arr = []
    body_count = 0
    # Loop over the keys of the dict.
    for key in data:
        if key.startswith(PAT_MOD) and key not in pplist:
            body_count += 1
            ret_arr.append(key)
            for k, v in conv_tens.items():
                if k in key and PAT_W in key and \
                    list(data[key].size()) == v[PAT_W]:
                    ret_arr.remove(key)
                    break
                elif k in key and PAT_B in key and \
                    list(data[key].size()) == v[PAT_B]:
                    ret_arr.remove(key)
                    break
    # On no mismatch set True.
    if not ret_arr:
        ret_val = True
    # Return True/False and mismatching data.
    return ret_val, ret_arr, body_count

# ------------------------
# Function check_pre_pos()
# ------------------------
def check_pre_pos(data):
    '''Check pre and post key/value pairs.'''
    ret_val = False
    ret_arr = []
    pp_count = 0
    for key in data:
        if key.startswith(PAT_MOD) and key in pplist:
            pp_count += 1
            ret_arr.append(key)
            if PAT_W in key:
                for k, v in ppdict.items():
                    if key == k and list(data[key].size()) == v:
                        ret_arr.remove(key)
                        break
            elif PAT_B in key:
                for k, v in ppdict.items():
                    if key == k and list(data[key].size()) == v:
                        ret_arr.remove(key)
                        break
    # On no mismatch set True.
    if not ret_arr:
        ret_val = True
    # Return True/False and mismatching data.
    return ret_val, ret_arr, pp_count

# -----------------------
# Function simple_check()
# -----------------------
def simple_check(model_content):
    # Perform some sipmple checks.
    weight_shape_ref = [64, 3, 3, 3]
    bias_shape_ref = [64]
    if weight_str_0 in model_content and bias_str_0 in model_content:
        weight_shape = list(model_content[weight_str_0].size())
        bias_shape = list(model_content[bias_str_0].size())
        if weight_shape == weight_shape_ref and bias_shape == bias_shape_ref:
            print(info_str_0)
        else:
            print(warn_str_0)
    elif weight_str_1 in model_content and bias_str_1 in model_content:
        weight_shape = list(model_content[weight_str_1].size())
        bias_shape = list(model_content[bias_str_1].size())
        if weight_shape == weight_shape_ref and bias_shape_ref == [64]:
            print(info_str_1)
        else:
            print(warn_str_1)
    else:
        # Print the content to the terminal window.
        print(model_content)
        print(error_str)
    # Return None
    return None

# -----------------------
# Function check_esrgan()
# -----------------------
def check_esrgan(model_data):
    '''Main script function'''
    # Perform some checks.
    print("\n***  key and tensor checks  ***")
    chkval0, chkarr0, cnt0 = check_keys(model_data)
    chkval1, chkarr1, cnt1 = check_tensors(model_data)
    chkval2, chkarr2, cnt2 = check_pre_pos(model_data)
    # Print not empty arrays.
    if chkarr0:
        print("\n{}".format(chkarr0))
    else:
        print("Key check ok. Noting to print out!")
    if chkarr1:
        print("\n{}".format(chkarr1))
    else:
        print("Tensor check ok. Noting to print out!")
    if chkarr2:
        print("\n{}".format(chkarr2))
    else:
        print("Pre/Post key check ok. Noting to print out!")
    # Check on mismatching lengths.
    if len(chkarr0) != cnt0 and len(chkarr0) > 0:
        print("\nMismatch in number of keys!")
        print(len(chkarr0), " wrong of ", cnt0)
    if len(chkarr1) != cnt1 and len(chkarr1) > 0:
        print("\nMismatch in number of body lines!")
        print(len(chkarr1), " wrong of ", cnt1)
    if len(chkarr2) != cnt2 and len(chkarr2) > 0:
        print("\nMismatch in number of pre/post keys!")
        print(len(chkarr2), " wrong of ", cnt2)
    if (chkval0 and chkval1 and chkval2) is True:
        info_str = "Old ESRGAN RRBD model. Perfekt match in the data structure."
        info_msg = "\n{0}{1}{2}".format(COL_GREEN, info_str, COL_DEFAULT)
        print(info_msg)
    elif (chkval0 and chkval1 and chkval2) is False:
        err_str = "NOT an Old ESRGAN RRBD model. No match in the data structure."
        err_msg = "\n{0}{1}{2}".format(COL_RED, err_str, COL_DEFAULT)
        print(err_msg)
    else:
        warn_str = "Maybe an Old ESRGAN RRBD model. Check the data!"
        warn_msg = "\n{0}{1}{2}".format(COL_YELLOW, warn_str, COL_DEFAULT)
        print(warn_msg)
    # Simple check.
    simple_check(model_data)
    # Return None
    return None

# ++++++++++++++++++++
# Main script function
# ++++++++++++++++++++
def main(filename):
    '''Main script function'''
    # print file type.
    print("***  file type  ***\n")
    print("File Type:", file_type(filename))
    # Try to load the model data from the file.
    # Must be of type collections.OrderedDict.
    try:
        model_content = torch.load(filename)
    except:
        err_str = "Could not load model data from file! Maybe not a valid model!"
        err_msg = "{0}{1}{2}".format(COL_RED, err_str, COL_DEFAULT)
        print(err_msg)
        return None
    # Check if it is dict or OrderedDict.
    if str(type(model_content)) == "<class 'collections.OrderedDict'>":
        pass
    elif str(type(model_content)) == "<class 'dict'>":
        key_var = list(model_content.keys())[0]
        if key_var == "params_ema" or key_var == "params":
            warn_str = "Take a look at the data structure. Maybe it is a RealESRGAN model!\33"
            warn_msg = "{0}{1}{2}".format(COL_CYAN, warn_str, COL_DEFAULT)
            print(warn_msg)
        else:
            warn_str = "Unknown model type. Take a look at the data structure!\33"
            warn_msg = "{0}{1}{2}".format(COL_CYAN, warn_str, COL_DEFAULT)
            print(warn_msg)
        # return None
        return None
    # Loop over they keys and print keyes ans tensor shape.
    print("\n***  keys and tensor shapes  ***\n")
    for key in model_content:
        dim = list(model_content[key].size())
        print(key, dim)
    # Call check function.
    check_esrgan(model_content)
    # Return None
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(model_name)
