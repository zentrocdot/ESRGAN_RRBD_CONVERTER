#!/usr/bin/python
#
# Usage:
# python3 RealESRGAN_test.py "shedevil.jpg" "RRDB_ESRGAN_x4_REAL.pth" "16" "gpu"
#
# Requirements:
# basicsr
# realesrgan

# Import the Python modules.
import warnings
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Import the third party Python modules.
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Get the arguments from the command line.
model_file_name = sys.argv[1]
upscaler = sys.argv[2]
outscale = sys.argv[3]
device = sys.argv[4]

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get basename and extension.
file_name = os.path.basename(model_file_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = "RRDB_" + basename + "_upscaled" + extension

# ------------
# Reset screen
# ------------
def reset_term() -> None:
    '''Reset the terminal window.'''
    ESC_SEQ = "\33c"
    sys.stdout.write(ESC_SEQ)
    sys.stdout.flush()
    return None

# ***********************************
# Function upscale_image_realesrgan()
# ***********************************
def upscale_image_realesrgan(filename, savename, upscaler, outscale, device):
    '''Upscale an image using RealESRGAN.'''
    # Memory mangement.
    fraction = 0.9
    torch.cuda.set_per_process_memory_fraction(fraction, device=None)
    # Free the GPU cache.
    torch.cuda.empty_cache()
    # Read the image from the file path.
    numpyImage = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # Check type of device.
    match device:
        case "CPU":
            device = torch.device('cpu')
        case "GPU":
            device = torch.device('cuda')
        case _:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check if outscale is None. If None set outscale to 4. Convert str to int.
    if outscale is None:
        outscale = 4
    else:
        outscale = int(outscale)
    # Set the netscale. Check which number for netscale should be used?
    netscale = 2
    # Set some other parameters. Check the parameter for future work.
    tile = 0
    gpu_id = None
    dni_weight = None
    tile_pad = 10
    pre_pad = 0
    accuracy = "fp16"
    # Set the model name / model path.
    model_name = upscaler
    # Set the scale variable for RRDBNet.
    scale = 4
    # Create the model with standard values.
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    # initialise the upsampler.
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_name,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=accuracy,
        gpu_id=gpu_id
        )
    # Finally upscale the image. upscaled is a numpy array.
    upscaled, _ = upsampler.enhance(numpyImage, outscale=outscale)
    # Convert upscaled image in BGR to image in RGB.
    upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    # Create an PIL image.
    image = Image.fromarray(upscaled)
    # Save the image.
    image.save(savename)
    # Return None.
    return None

# +++++++++++++
# Main function
# +++++++++++++
def main(file_name, save_name, upscaler, outscale, device):
    upscale_image_realesrgan(file_name, save_name, upscaler, outscale, device)
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    upscale_image_realesrgan(file_name, save_name, upscaler, outscale, device)
