# ESRN RRBG CONVERTER [![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN)

## Preface

<p align="justify">While I was working on the upscaling method for
images using ESRGAN models, the question arose as to how I could
utilise all available ESRGAN models with the approach I am working
with.</p>

<p align="justify">If you use ESRGAN models from external sources,
you will get sometimes RRBD error messages indicating a tensor
formatting problem.</p>

<p align="justify">A simple converter can be found in the original 
ESRGAN sources. I have adapted this converter for my personal 
purposes.</p>

<p align="justify">Now upscalers that I can use with <i>AUTOMATIC1111</i>
can be used. I can now also use upscalers from other sources.</p>

## Motivation

<p align="justify">I implemneted ERSGAN [1] and RealESRGAN [2] in my
<i>Lazy Image Upscaler</i>. To be able to test more than the given 4 
models from xinntao I searched and collected other ERSGAN models and
tried them out, most of the time with no success. The Model from 
AUTOMATIC1111 [5] was also working. The usage of other models from other 
sources failed.</p>

<p align="justify">I started a mini project with the goal to provide
other upscaler models for my <i>Lazy Image Upscaler</i>. With the here
presented converter I have now access to more models.</p>

<p align="justify">I will use the ERSGAN method in his given form from
xinntao independend from other software tools. The only thing I need is
a converter to prepare more or less the most ESRGAN models for use
with xinntaos approach.</p>

## How to Use the Converter

<p align="justify">You need following two files from the scripts folder.</p>

```
RRDBNet_arch.py
```

```
converter_RRDB_models.py
```

<p align="justify"><code>converter_RRDB_models.py</code> is the converter and <code>RRDBNet_arch.py</code> contains the classes
which are required to run the converter. The latter file is imported from the converter.</p>

<p align="justify">Run the converter as follows:</p>

```
python3 converter_RRDB_models.py 4x_Superscale-SP8000G.pth
```

## Error handling

<p align="justify">Errors are catched and the Tracback is printed out into the Terminal window. This looks like:
</p>

```
Traceback (most recent call last):
  File "/home/hades/ssd-sandisk/AI_Tools/ESRGAN/ESRGAN/models/experimentell_converter_RRDB_models.py", line 97, in main
    crt_net[key] = pretrained_net[value]
KeyError: 'model.8.weight'

Traceback (most recent call last):
  File "/home/hades/ssd-sandisk/AI_Tools/ESRGAN/ESRGAN/models/experimentell_converter_RRDB_models.py", line 97, in main
    crt_net[key] = pretrained_net[value]
KeyError: 'model.8.bias'

```

## Tested Models
 
+ 4x_foolhardy_Remacri.pth
+ 4xPSNR.pth
+ RRDB_ESRGAN_x4.pth
+ 4x-UltraSharp.pth

## ESRGAN and AUTOMATIC1111

<p align="justify">AUTOMATIC is using the ESRGAN model which can be downloaded from [5].</p>

## To-Do

<p align="justify">I have to check under which conditions the converter
is working and under which conditions the converter is not working.</p>

<p align="justify">Analysis of the internal model structure to understand
the formatting of different models related to RRDB models.</p>
 
<p align="justify">I have to catch errors while executing the script.
In the experimental version I was able to realise this in a rudimentary
way.</p>
 
<p align="justify">I need an improvement of the comments in the script.
And I am also need an improvement of this documentation.</p>

## Reference

[1] https://github.com/xinntao/ESRGAN

[2] https://github.com/xinntao/Real-ESRGAN

[3] https://huggingface.co/

[4] https://openmodeldb.info/

[5] https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth

[6] https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY

[7] https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ

