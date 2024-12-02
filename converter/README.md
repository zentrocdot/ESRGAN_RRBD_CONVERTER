# ESRGAN CONVERTER [![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN) [![GitHub  - RealEsrGAN](https://img.shields.io/badge/GitHub_-RealESRGAN-9933ff)](https://github.com/xinntao/Real-ESRGAN)

## Preface

<p align="justify">I am writing this README section to clarify some
things for myself. Since I am absolutely new to the field of creating
and using AI models, there are some basic things to clarify.</p>

## CONVERTERS

<p align="justify">The names of the converters are self-explanatory.</p>

- oldESRGAN_to_newESRGAN_converter.py
- ewESRGAN_to_RealESRGAN_converter.py
- RealESRGAN_to_newESRGAN_converter.py

<p align="justify">Usage:</p>

```
oldESRGAN_to_newESRGAN_converter.py <upscaler_model.pth>
newESRGAN_to_RealESRGAN_converter.py <upscaler_model.pth>
RealESRGAN_to_newESRGAN_converter.py <upscaler_model.pth>

```

# TL;DR

## Possible Conversions

<p align="justify">I am currently considering two scenarios for the
converters.</p>

<p align="justify">The directions of the conversions are shown by
the arrows.</p>

+ Scenario I

```
old ESRGAN → new ESRGAN → RealESRGAn
```

+ Scenario II

```
RealESrgan →  new ESRGAN
```

## Why Are This Converters Working

<p align="justify">I refer to new ESRGAN as a 
reference for all considerations</p>

<p align="justify">All explanations refer to new
ESRGAN. The other explanations for the other models 
can be transferred or derived.</p>

<p align="justify">I wrote two test scripts to show
how it works. The test scripts can be found in the 
folder <code>test_scripts</code>.</p>

<p align="justify">The first script creates a model.
Then I am extracting the state dict of the model.
Afterwards I am printing the state dict of this model
in form of key/value as skeleton. Next I am loading
in a second script the available upscaler. Then I am
printing the content (state dict) of this model in
form of key/value as skeleton. I am writing both 
results in a text file and both files are identical.

```
python3 ESRGAN_RRDB_model.py > test1.txt
python3 ESRGAN_RRDB_model_test.py > test2.txt

diff -s test1.txt test2.txt

results in

Files test1.txt and test2.txt are identical.

```

## How the Converter Works

<p align="justify">Each converter simply changes the 
literal representation of each key with respect to the
related model.</p>

## Printout of a State Dict

<p align="justify">We can distinguish between three sections.
This are two header line, the body with weight/bias and the
ten footer line.</p>

<p align="justify">The printout is still a skeleton consisting
of key and value in form of a tensor with the size/shape of the
tensor. The internal tensor data are omitted. The spaces and the
vertical dots are added for a better visibility. The key items
are numbered in the full version from 0 to 22.</p>

```
conv_first.weight                ->  tensor([...], size=(64, 3, 3, 3))
conv_first.bias                  ->  tensor([...], size=(64,))

RRDB_trunk.0.RDB1.conv1.weight   ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.0.RDB1.conv1.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB1.conv2.weight   ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.0.RDB1.conv2.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB1.conv3.weight   ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.0.RDB1.conv3.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB1.conv4.weight   ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.0.RDB1.conv4.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB1.conv5.weight   ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.0.RDB1.conv5.bias     ->  tensor([...], size=(64,))
RRDB_trunk.0.RDB2.conv1.weight   ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.0.RDB2.conv1.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB2.conv2.weight   ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.0.RDB2.conv2.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB2.conv3.weight   ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.0.RDB2.conv3.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB2.conv4.weight   ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.0.RDB2.conv4.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB2.conv5.weight   ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.0.RDB2.conv5.bias     ->  tensor([...], size=(64,))
RRDB_trunk.0.RDB3.conv1.weight   ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.0.RDB3.conv1.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB3.conv2.weight   ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.0.RDB3.conv2.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB3.conv3.weight   ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.0.RDB3.conv3.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB3.conv4.weight   ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.0.RDB3.conv4.bias     ->  tensor([...], size=(32,))
RRDB_trunk.0.RDB3.conv5.weight   ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.0.RDB3.conv5.bias     ->  tensor([...], size=(64,))

⋮

RRDB_trunk.22.RDB1.conv1.weight  ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.22.RDB1.conv1.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB1.conv2.weight  ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.22.RDB1.conv2.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB1.conv3.weight  ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.22.RDB1.conv3.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB1.conv4.weight  ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.22.RDB1.conv4.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB1.conv5.weight  ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.22.RDB1.conv5.bias    ->  tensor([...], size=(64,))
RRDB_trunk.22.RDB2.conv1.weight  ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.22.RDB2.conv1.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB2.conv2.weight  ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.22.RDB2.conv2.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB2.conv3.weight  ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.22.RDB2.conv3.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB2.conv4.weight  ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.22.RDB2.conv4.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB2.conv5.weight  ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.22.RDB2.conv5.bias    ->  tensor([...], size=(64,))
RRDB_trunk.22.RDB3.conv1.weight  ->  tensor([...], size=(32, 64, 3, 3))
RRDB_trunk.22.RDB3.conv1.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB3.conv2.weight  ->  tensor([...], size=(32, 96, 3, 3))
RRDB_trunk.22.RDB3.conv2.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB3.conv3.weight  ->  tensor([...], size=(32, 128, 3, 3))
RRDB_trunk.22.RDB3.conv3.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB3.conv4.weight  ->  tensor([...], size=(32, 160, 3, 3))
RRDB_trunk.22.RDB3.conv4.bias    ->  tensor([...], size=(32,))
RRDB_trunk.22.RDB3.conv5.weight  ->  tensor([...], size=(64, 192, 3, 3))
RRDB_trunk.22.RDB3.conv5.bias    ->  tensor([...], size=(64,))

trunk_conv.weight                ->  tensor([...], size=(64, 64, 3, 3))
trunk_conv.bias                  ->  tensor([...], size=(64,))
upconv1.weight                   ->  tensor([...], size=(64, 64, 3, 3))
upconv1.bias                     ->  tensor([...], size=(64,))
upconv2.weight                   ->  tensor([...], size=(64, 64, 3, 3))
upconv2.bias                     ->  tensor([...], size=(64,))
HRconv.weight                    ->  tensor([...], size=(64, 64, 3, 3))
HRconv.bias                      ->  tensor([...], size=(64,))
conv_last.weight                 ->  tensor([...], size=(3, 64, 3, 3))
conv_last.bias                   ->  tensor([...], size=(3,))
```

## References

[1] https://github.com/xinntao/ESRGAN

[2] https://github.com/xinntao/Real-ESRGAN

[3] https://pytorch.org/docs/stable/generated/torch.load.html

[4] https://pytorch.org/tutorials/beginner/saving_loading_models.html

[5] https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

[6] https://pytorch.org/docs/stable/generated/torch.nn.Module.html
