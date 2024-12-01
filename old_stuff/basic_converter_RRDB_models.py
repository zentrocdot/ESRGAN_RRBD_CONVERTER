#!/usr/bin/python3
#
# Version 0.0.0.1
#
# (C) 2018 xinntao
# (C) 2024 zentrocdot

# Import the Python modules.
import os
import sys
import torch
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
save_name = basename + "_rrdb" + extension

# Print input and output name.
print("Input:", file_name)
print("Output:", save_name)

# Start conversion.
print("Start conversion ...")

# Set pretrained net name.
pretrained_net = torch.load(file_name)

# Model conversion.
crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
crt_net = crt_model.state_dict()

load_net_clean = {}
for k, v in pretrained_net.items():
    if k.startswith('module.'):
        load_net_clean[k[7:]] = v
    else:
        load_net_clean[k] = v
pretrained_net = load_net_clean

tbd = []
for k, v in crt_net.items():
    tbd.append(k)

for k, v in crt_net.items():
    if k in pretrained_net and pretrained_net[k].size() == v.size():
        crt_net[k] = pretrained_net[k]
        tbd.remove(k)

crt_net['conv_first.weight'] = pretrained_net['model.0.weight']
crt_net['conv_first.bias'] = pretrained_net['model.0.bias']

for k in tbd.copy():
    if 'RDB' in k:
        ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
        if '.weight' in k:
            ori_k = ori_k.replace('.weight', '.0.weight')
        elif '.bias' in k:
            ori_k = ori_k.replace('.bias', '.0.bias')
        crt_net[k] = pretrained_net[ori_k]
        tbd.remove(k)

crt_net['trunk_conv.weight'] = pretrained_net['model.1.sub.23.weight']
crt_net['trunk_conv.bias'] = pretrained_net['model.1.sub.23.bias']
crt_net['upconv1.weight'] = pretrained_net['model.3.weight']
crt_net['upconv1.bias'] = pretrained_net['model.3.bias']
crt_net['upconv2.weight'] = pretrained_net['model.6.weight']
crt_net['upconv2.bias'] = pretrained_net['model.6.bias']
crt_net['HRconv.weight'] = pretrained_net['model.8.weight']
crt_net['HRconv.bias'] = pretrained_net['model.8.bias']
crt_net['conv_last.weight'] = pretrained_net['model.10.weight']
crt_net['conv_last.bias'] = pretrained_net['model.10.bias']

torch.save(crt_net, save_name)

print("... conversion completed!")
