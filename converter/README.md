# ESRGAN CONVERTER [![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN) [![GitHub  - RealEsrGAN](https://img.shields.io/badge/GitHub_-RealESRGAN-9933ff)](https://github.com/xinntao/Real-ESRGAN)

## Preface

<p align="justify">I am writing this README section to clarify some
things for myself. Since I am absolutely new to the field of creating
and using AI models, there are some basic things to clarify.</p>

## CONVERTERS

<p align="justify">The names of the converters are self-explanatory.</p>

+ oldESRGAN_to_newESRGAN_converter.py
+ newESRGAN_to_RealESRGAN_converter.py
+ RealESRGAN_to_newESRGAN_converter.py

<p align="justify">Usage:</p>

```
+ oldESRGAN_to_newESRGAN_converter.py <upscaler_model.pth>
+ newESRGAN_to_RealESRGAN_converter.py <upscaler_model.pth>
+ RealESRGAN_to_newESRGAN_converter.py <upscaler_model.pth>

```

# TL;DR

## Conversions

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

I wrote two test scripts to show how it works.


I refer to new ESRGAN as a reference for all considerations

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

[3] 
