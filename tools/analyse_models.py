#!/usr/bin/python
'''Structured reading data from .pth file.'''
#
# Analyse Model Data
# Version 0.0.0.1
#
# pylint: disable=useless-return
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

# Import standard Python modules.
import sys
import warnings

# Import third party module.
import torch

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set torch print options.
torch.set_printoptions(threshold=10_000)

# Get the model name from the command line.
model_file_name = sys.argv[1]

# Reset screen
def reset_term() -> None:
    '''Reset the terminal window.'''
    ESC_SEQ = "\33c"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# Clear screen
def clear_term() -> None:
    '''Clear the terminal window.'''
    ESC_SEQ = "\33[2J\33[H"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# Main script function.
def main(filename):
    '''Main script function'''
    # Set some key strings.
    error_str = "\n\33[41mERROR: Check the data structure!\33[49m"
    warn_str_0 = "\n\33[46mFound UNKNOWN (OLD) ESRGAN RRBD model. Check the data!\33[49m"
    warn_str_1 = "\n\33[46mFound UNKNOWN (NEW) ESRGAN RRBD model. Check the data!\33[49m"
    info_str_0 = "\n\33[45mFound old ESRGAN RRBD model\33[49m"
    info_str_1 = "\n\33[42mFound new ESRGAN RRBD model\33[49m"
    weight_str_0 = 'model.0.weight'
    bias_str_0 = 'model.0.bias'
    weight_str_1 = 'conv_first.weight'
    bias_str_1 = 'conv_first.bias'
    # Load the file content into data.
    # Is of type collections.OrderedDict.
    model_content = torch.load(filename)
    # Loop over the keys in model content.
    for key in model_content:
        try:
            # Get shape of tensor.
            dim = list(model_content[key].size())
            # print key and shape.
            print(key, dim)
        except:
            pass
    # Perform some sipmple checks.
    weight_shape_ref = [64, 3, 3, 3]
    bias_shape_ref = [64]
    if (weight_str_0 and bias_str_0) in model_content:
        weight_shape = list(model_content[weight_str_0].size())
        bias_shape = list(model_content[bias_str_0].size())
        if weight_shape == weight_shape_ref and bias_shape == bias_shape_ref:
            print(info_str_0)
        else:
            print(warn_str_0)
    elif (weight_str_1 and bias_str_1) in model_content:
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

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(model_file_name)
