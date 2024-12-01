#!/usr/bin/python
#
# Usage:
# python3 ESRGAN_test.py "shedevil.jpg" "RRDB_ESRGAN_x4_REAL.pth" "gpu"
#
# Requirements:
# RRDBNet_arch

# Import the Python modules.
import warnings
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Import Python module.
import RRDBNet_arch as arch

# Get the arguments from the command line.
model_file_name = sys.argv[1]
upscaler = sys.argv[2]
device = sys.argv[3]

# Ignore (torch) future warning.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Get basename and extension.
file_name = os.path.basename(model_file_name)
fn_list = os.path.splitext(file_name)
basename = fn_list[0]
extension = fn_list[1]

# Create the save name.
save_name = basename + "_upscaled" + extension

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
# Function upscale_image_esrgan()
# ***********************************
def upscale_image_realesrgan(filename, savename, upscaler, device):
    '''Upscale an image using RealESRGAN.'''
    # Memory mangement.
    fraction = 0.9
    torch.cuda.set_per_process_memory_fraction(fraction, device=None)
    # Free the GPU cache.
    torch.cuda.empty_cache()
    # Check type of device.
    match device:
        case "CPU":
            device = torch.device('cpu')
        case "GPU":
            device = torch.device('cuda')
        case _:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get the model path from dict.
    model_path = upscaler
    # Initalise the model.
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    try:
        model.load_state_dict(torch.load(model_path), strict=True)
    except:
        print("Could not load the state dictianary of the model. Leaving Upscaler!")
        return None
    model.eval()
    # Move model to device.
    model = model.to(device)
    # Read the image from the file path.
    numpyImage = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # 8-bit image to floating-point image.
    image = numpyImage * 1.0 / 255
    # Transpose the numpy array.
    image = np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))
    # Create a tensor.
    image = torch.from_numpy(image).float()
    # Flatten the tensor.
    image_LR = image.unsqueeze(0)
    # Move image to device.
    image_LR = image_LR.to(device)
    # Interpolate the new upscaled image.
    with torch.no_grad():
        output = model(image_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # Transpose the numpy array.
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    # Create the upscaled image.
    upscaled = (output * 255.0).round()
    # Convert the upscaled numpy image from BGR to RGB.
    upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    # Create an PIL image from the numpy image.
    image = Image.fromarray(upscaled.astype('uint8'))
    # Save the PIL image.
    image.save(savename)
    # Return None.
    return None

# +++++++++++++
# Main function
# +++++++++++++
def main(file_name, save_name, upscaler, device):
    upscale_image_realesrgan(file_name, save_name, upscaler, device)
    return None

# Execute as module as well as a program.
if __name__ == "__main__":
    # Reset terminal.
    reset_term()
    # Call the main function.
    upscale_image_realesrgan(file_name, save_name, upscaler, device)
