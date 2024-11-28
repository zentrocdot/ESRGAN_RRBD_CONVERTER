# ESRN RRBG CONVERTER [![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN)

## Preface

<p align="justify">While I was working on implementing an upscaling
method for images using ESRGAN models, the question arose as to how
I can use all available ESRGAN models with the approach I am working
with.</p>

<p align="justify">If one uses ESRGAN models from external sources,
one will get sometimes RRBD error messages indicating a tensor
formatting problem.</p>

<p align="justify">A simple converter can be found in the original 
ESRGAN sources I intended to use. I have adapted this converter for
my personal purposes.</p>

<p align="justify">Now some upscalers that I am using with
<i>AUTOMATIC1111</i> I can also use in my upscaler application.
I can now also use upscalers from other sources.</p>

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

<p align="justify">You need following two files from the <code>scripts</code>
folder.</p>

```
RRDBNet_arch.py
```

```
converter_RRDB_models.py
```

<p align="justify"><code>converter_RRDB_models.py</code> is the
converter script and <code>RRDBNet_arch.py</code> is the module
which contains the classes which are required to run the converter
script. The latter file is imported from the converter.</p>

<p align="justify">Run the converter as follows:</p>

```
python3 converter_RRDB_models.py upscaler_model_file_name.pth
```

## Error Handling

<p align="justify">Errors are catched and the Tracback is printed out into the terminal window. This looks like:
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

<p align="justify">Other errors need to be catched also.</p>

## Tested Models

<p align="justify">ESRGAN models I have tested and successful converted:</p>
 
+ 4x_foolhardy_Remacri.pth
+ 4xPSNR.pth
+ 4x-UltraSharp.pth
+ 4xLSDIRplus.pth
+ RRDB_ESRGAN_x4.pth
+ 4xPSNR.pth
+ 4x_Fatality_Comix_260000_G_rrdb.pth

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

## Test Environment

<p align="justify">I developed and tested the Python script with following specification:</p>

* Linux Mint 21.3 (Virginia)
* Python 3.10.14
* OpenCV 4.10.0
* PIL 11.0.0
* Gradio 5.6.0
* Torch 2.4.1+cu121
* Numpy  2.1.3

## Reference

[1] https://github.com/xinntao/ESRGAN

[2] https://github.com/xinntao/Real-ESRGAN

[3] https://huggingface.co/

[4] https://openmodeldb.info/

[5] https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth

[6] https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY

[7] https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ

## Donation

<p align="justify">If you like what I present here, or if it helps you,
or if it is useful, you are welcome to donate a small contribution. Or
as you might say: Every TRON counts! Many thanks in advance! :smiley:
</p>

> ###### <p align="left">Crypto Coin Tron</p>

```
TQamF8Q3z63sVFWiXgn2pzpWyhkQJhRtW7
```

> ###### <p align="left">Crypto Coin Dogecoin</p>

```
DQYkNGW8VfCuUbM9Womnp6KiFdtMa4NUkD
```

> ###### <p align="left">Crypto Coin Bitcoin</p>

```
12JsKesep3yuDpmrcXCxXu7EQJkRaAvsc5
```

> ###### <p align="left">Crypto Coin Ethereum</p>

```
0x31042e2F3AE241093e0387b41C6910B11d94f7ec
```
<p align="left">$${\textnormal{\color{purple}Have a nice and successful day. And peace to everyone on earth.}}$$</p>
