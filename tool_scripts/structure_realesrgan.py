#!/usr/bin/python
'''Print keys and shapes.'''
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
    # Load the file content into data.
    model_data = torch.load(filename)
    print("\n", type(model_data), "\n")
    # Print the raw content.
    print(model_data, "\n")
    # Get the keyword from list (params_ema or params).
    key_word = list(model_data.keys())[0]
    print(key_word)
    # Extract the pretrained model
    pretrained_net = model_data[key_word]
    print("\n", type(pretrained_net), "\n")
    # Loop over the keys of the pretrained model.
    for key in pretrained_net:
        # Get shape of tensor.
        dim = list(pretrained_net[key].size())
        # print key and shape.
        print(key, dim)
    # Return None
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    main(model_file_name)
