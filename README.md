# ESRGAN RRBG CONVERTER [![Contact - Discord](https://img.shields.io/badge/Contact-Discord-ffcc00)](https://discord.gg/bHFQfTwfAJ)
[![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN) [![GitHub  - RealEsrGAN](https://img.shields.io/badge/GitHub_-RealESRGAN-9933ff)](https://github.com/xinntao/Real-ESRGAN) [![Models - OpenModelDB](https://img.shields.io/badge/Contact-OpenModelDB-0033cc)](https://openmodeldb.info/?q=ESRGAN) [![Models - Hugging Face](https://img.shields.io/badge/Contact-Hugging_Face-ff6600)](https://huggingface.co/models?sort=trending&search=ESRGAN)

### SHORT DESCRIPTION OF the REPOSITORY

<p align="justify">Resources for converting of models from old 
ESRGAN to new ESRGAN architecture. Resources for converting 
RealESRGAN models to new ESRGAN models. There is also a converter
for converting new ESRGAN models to RealESRGAN models. These
converters can be found in folder <code>converter</code>.
Bonus material are the standalone upscalers for ESRGAN and
RealESRGAN (The latter two are rudimentarily documented). Last
standalone converter versions can be found in folder
<code>upscaler</code>. For the other stuff read the 
documentation.</p>

---

### CONTRIBUTION

<b><p align="justify">If you like what I present here, or if it 
helps you, or if it is useful, you are welcome to [donate](#Donation)
a small contribution. It motivates me a lot and speeds up my work
a much 😏.</p></b>

---

> [!NOTE]
> <p align="justify">All scripts are programmed for use with the Linux
> operating system. I do not consider other operating systems due to
> time constraints. The software used is freely available and makes
> no special demands on the operating system.</p>

# TL;DR: Use the Table of Contents for Topics of Interest

## Preface

<p align="justify">A small idea has now turned into a small project.
To make myself independent of the web user interfaces for the AI image
generation, which are offering a high sophisticated upscaling capability,
I implement different approaches for the upscaling of images for myself.</p>

<p align="justify">I worked in parallel with ESRGAN and RealESRGAN.
Both approaches delivered quite impressive results from a standing start. 
However, the number of functioning models is limited for the time being. 
This is where the converters come into play, with which the number of
models can potentially be increased. 
</p>

<p align="justify">At short notice and on a whim, I took a closer and
more intensive look at the topic. The interim result of my deliberations
can be found here in the repository. But only some of the things I have
tried and done can be found here.</p>

<p align="justify">The quality of the results achieved in the meantime
speaks for itself and justifies the use of ESRGAN and RealESRGAN.</p>

## Motivation

<p align="justify">While I was working on implementing an upscaling
method for images using ESRGAN models, the question arose as to how
I can use all available ESRGAN models with the approach I am working
with.</p>

<p align="justify">If one uses ESRGAN models from external sources 
and not the original proposed ESRGAN models, one will get sometimes 
errors which results in a failure of the upscaling process.</p>

<p align="justify">A simple converter could be found in the original 
ESRGAN sources I intended to use. That this is a converter was not
obvious.I have adapted this converter for my personal purposes.</p>

<p align="justify">Now some upscalers that I am already using with
the web user interface <i>AUTOMATIC1111</i> I can also use in my
own upscaler application or standalone. I can now also use upscalers
from different other sources.</p>

## Introduction

<p align="justify">I implemented ERSGAN [1] and RealESRGAN [2] in
my <i>Lazy Image Upscaler</i>. To be able to test more than the given
four models from xinntao I searched and collected other ERSGAN models
and tried them out, most of the time without success. One model in use
by the web user interface <i>AUTOMATIC1111</i> [5] was working out of 
the box. The usage of other models from other sources failed for the
time being.</p>

<p align="justify">I will use the ERSGAN method (and the RealERSGAN) in
his given form from xinntao independend from other software tools. The 
only thing I need is a converter to prepare more or less the most ESRGAN
models for use with xinntaos approach.</p>

> [!NOTE]  
> I am interested in the generative AI image creation. Of course, I am
> also interested in theory when necessary, but only when it is helpful.
> Keeping this in mind my interest is focussed on how to use a model and
> not how to train a model.

## Name of the Repository

<p align="justify">ESRGAN is used together with RRDB. This is
the base concept. I provide tools for the conversion between
ESRGAN models and converter for this methodology. So after
some back and forth the name was born. At the latest when
I understood the connections better.</p>

## Technical Background

### Brief Introduction

<p align="justify">The ESRGAN (Enhanced Super-Resolution Generative
Adversarial Networks) is improving the model architecture using the
RRDB (Residual-in-residual Dense Block) without batch normalization
based on the observations of EDSR. ESRGAN uses RaGAN (Relativistic
GAN) relative loss instead of the perceptual loss and adversarial
loss. It also improves the perceptual loss with VGG loss used in
SRGAN. ESRGAN and RealESRGAn are direct improvments of SRGAN. The
influence of EDSR should also be noted. A good article to the topic
is [8].</p>

Abbreviations explained can be found [here](https://civitai.com/articles/7432/ai-abbreviations-uncovered).

### ESRGAN & RealESRGAN

<p align="justify">Within this repository I am focussing on ESRGAN
and RealESRGAN. When we are talking about the implmentation of what 
the last sections explains, we have to discuss the internal structure 
of given ESRGAN model.</p>

<p align="justify">The old (outdated) models have a different internal
structure to the new (current) models with respect to the used keys.
The value to the key is still a tensor.</p>

<p align="justify">For the conversion one needs a translation table
or conversion table for the keys which looks like:</p>

```
    "model.0.weight"                     ->  "conv_first.weight"
    "model.0.bias"                       ->  "conv_first.bias":
    "model.1.sub.0.RDB1.conv1.0.weight"  ->  "RRDB_trunk.0.RDB1.conv1.weight" 
    "model.1.sub.0.RDB1.conv1.0.bias"    ->  "RRDB_trunk.0.RDB1.conv1.bias"
    "model.1.sub.23.weight"              ->  "trunk_conv.weight"
    "model.1.sub.23.bias"                ->  "trunk_conv.bias" 
    "model.3.weight"                     ->  "upconv1.weight" 
    "model.3.bias"                       ->  "upconv1.bias" 
    "model.6.weight"                     ->  "upconv2.weight" 
    "model.6.bias"                       ->  "upconv2.bias" 
    "model.8.weight"                     ->  "HRconv.weight" 
    "model.8.bias"                       ->  "HRconv.bias" 
    "model.10.weight"                    ->  "conv_last.weight"
    "model.10.bias"                      ->  "conv_last.bias"
```

<p align="justify">The value related to the key is a tensor.
One has to consider while converting that given tensor shape
is the tensor shape which is required by the final model.</p>

<p align="justify">RealESRGAN has its own naming convention, which is
similar but not the same. Within the models that are compatible with 
the approach of xinntao, I can easily convert back and forth. I found
further differences in other ESRGAN models that require a further
investigation.</p>

## Description of What Is Implemented So Far

<p align="justify">I have converted the original supplied converter
for old RSGAN to the new RSGAN for my personal purposes. Then I 
modernized the implementation a little bit. Next I figured out, that
some models look like ESRGAN models, but that they are not the models 
I can use or that they are no ESRGAN or somes´thing is strange with
the models. To have an idea what a model does I wrote a simple 
analysing tools. This tool identifies highly reliable old an new
ERSGAN models and it is able to find out if a model is possibly
RealsESRGAN. This is important driven by the fact that sometime 
a model is wrong declared. Next I wrot a converter from RealESRGAN
to ESRGAN. This converter works so far quite good.</p>

## Testing

<p align="justify">For testing purpose I wrote an upscaler for
ESRGAN and an upscaler for RealESRGAN. Both are upscaling an
image and simply save them.</p>

<p align="justify">I used these two upscalers next to an
experimental version of my <i>Lazy Image Upscaler</i> to
check that the converter are working the right way.</p>

## Easy Way to Test the Converter

### Installation

Clone this github repository and move into the directory 

```
git clone https://github.com/zentrocdot/ESRGAN_RRBD_CONVERTER.git
cd ESRGAN_RRDB_CONVERTER
```

or use the zip-file or simply download the files, which you need.

### How to Use the Main OLD Converter

> [!NOTE]
> <p align="justify">The following is the old description. Go to the folder
> <code>converter</code> for the new converters and descriptions.</p>

<p align="justify">You need following two files from the <code>old_stuff</code>
folder to get the converter run.</p>

```
RRDBNet_arch.py
```

```
converter_RRDB_models.py
```

<p align="justify"><code>converter_RRDB_models.py</code> is the
converter script and <code>RRDBNet_arch.py</code> is the Python 
module which contains the classes which are required to run the
converter script. The latter file is imported from the converter.
</p>

<p align="justify">Run the converter as follows:</p>

```
python3 converter_RRDB_models.py <upscaler_model_file_name.pth>
```

## Conversion Process

<p align="justify">If you run the experimental converter.</p>

```
lucifer@hades:~/ESRGAN/models$ python3 experimenta_converter_RRDB_models.py 8x_NMKD-Superscale_150000_G.pth
```

<p align="justify">the output in the terminal window is as follows:</p>

```
***  ESRGAN CONVERTER  ***
Reference keywords to found keywords:
['model.8.weight', 'model.8.bias', 'model.10.weight', 'model.10.bias']
['model.11.weight', 'model.11.bias', 'model.13.weight', 'model.13.bias']
Input: 8x_NMKD-Superscale_150000_G.pth
Output: 8x_NMKD-Superscale_150000_G_CED.pth
Start conversion ...
... conversion completed!
```

## Error Handling

<p align="justify">Errors are catched and the <i>Traceback</i>
is printed out into the terminal window. This looks like:</p>

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

<p align="justify">This is no longer an error. This behaviour
is considered in the last version of the converter. But other
errors need to be catched.</p>

<p align="justify">Reference keywords versus found keywords show,
if there was a mismatch.</p>

## Compatible NEW (current) ESRGAN Models

<p align="justify">Some were analysed and found to be
new models. No need for a conversion:</p>

* RRDB_ESRGAN_x4.pth (xinntao)
* RRDB_PSNR_x4.pth (xinntao)
* ESRGAN.pth (KAIR)
* DF2K.pth

## Converted Models from Old ESRGAN to New ESRGAN

<p align="justify">The focus of these repositories lies on ESRGAN
and everything related to it. Later, I will look into how I can
take RealESRGAN into account in an appropriate way.</p>

<p align="justify">The old ESRGAN models are listed below which
I have tested and successfully converted to new ESRGAN models:</p>
 
+ 4xFSMangaV2_CVTD.pth
+ 4xLSDIRplus.pth
+ 4xNomos2_otf_esrgan.pth
+ 4xNomosWebPhoto_esrgan.pth
+ 4xNomos8kSC.pth
+ 4xFSDedither.pth
+ 4x-eula-digimanga-MiA-65k.pth (Color reduction ???)
+ 4x-UltraSharp.pth
+ 4x-UniScale_Restore.pth
+ 4x-UniScaleV2_Sharp.pth
+ 4x_BooruGan_600k_CVTD
+ 4x_Fatality_Comix_260000_G.pth
+ 4x_foolhardy_Remacri.pth
+ 4x_FuzzyBox.pth
+ 4x_IllustrationJaNai_V1_ESRGAN_135k_CVTD
+ 4x_NMKD-Siax_175k.pth
+ 4x_NMKD-Siax_200k.pt
+ 4x_UniversalUpscalerV2-Sharp_101000_G.pth
+ 4xPSNR.pth
+ 8xPSNR.pth
+ 8x_NMKD-Superscale_150000_G.pth (color shift after conversion ???)
+ realesrgan-x4minus.pth
+ RRDB_PSNR_x4_old_arch.pth (xinntao)
+ RRDB_ESRGAN_x4_old_arch.pth (xinntao)

<p align="justify">After conversion I tried out all of
the listed models. When there was a problem it is noted
in the round brackets.</p>

<p align="justify">The list is not complete, but shows
that almost every old model can already be converted. 
The algorithm seems to work well so far.</p>

## ESRGAN and AUTOMATIC1111

<p align="justify">AUTOMATIC is using the ESRGAN model
which can be downloaded from [5].</p>

## Practical Application Example

<p align="justify">I used the following AI created image to show how
the conversion from one model to another model works. The original jpg
image has a resolution of 512 x 512 pixel.</p>

<a target="_blank" href=""><img src="./images/shedevil.jpg" alt="button panel"></a>

<i>Figure 1: Original image with a resolution of 512 x 512 pixel</i>

<a target="_blank" href=""><img src="./images/out_fill.jpg" alt="button panel"></a>

<i>Figure 2: Original image with marked area with a resolution of 512 x 512 pixel</i>

<p align="justify">I used the old ESRGAN model 
<code>RRDB_PSNR_x4_old_arch.pth</code> from xinntao 
and converted this model to a new ESRGAN model. I used 
the RealESRGAN model <code>RRDB_RealESRGAN_x4plus.pth</code> from 
xinntao and converted this model to a new ESRGAN model.</p>

<p align="justify">One can see is the following collage, what the theory
to the both approaches states. There are upscaling artifact in the ESRGAN
created image and no longer upscaling artifacts in the RealSRGAN created
image. I zoomed in the face of the girl in the created images to show the
effects.</p> 

<a target="_blank" href=""><img src="./images/collage.jpg" alt="button panel"></a>

<i>Figure 3: Collage of the cropped area of the original image with 512 x 512 pixel and the upscaled images with 2048 x 2048 pixel</i>

## Advantages and Disadvantages

## ESRGAN

<p align="justify">The disadvantage of ESRGAN is the fact that the models are
restircted to an upscaling of factor 4. The advantage is that there are a lot
of models which can be used.</p>

## RealESRGAN

<p align="justify">The disadvantage of ESRGAN is the fact that there are are a 
limited number of models which can be used. The advantage is the fact that in 
principle an upscaling with an arbitrary scale factor is possible.</p>

## Pickle Tensor

<p align="justify">Files in the Pickle Tensor fileformat have
in case of the ESRGAN models the extension <code>.pth</code>.
Binary as well as zip-files can be used. There is no need for
a further distinction or conversion.</p>
 
## Repository & Directory Structure

<p align="justify">The repository and directory structure of
the <i>ESRGAN RRDB CONVERTER</i> is looking as follows:</p> 

```bash
    └── esrgan_rrbd_converter
        ├── images
        ├── converter
        ├── upscaler
        ├── old_stuff
        ├── original_files
        ├── tool_scripts
        ├── test_scripts
        └── helper_scripts
```

<p align="justify">The folder <code>original_files</code> contains the original 
sources from xinntao [1]. In the folder <code>tool_scripts</code> there are tools 
like the one for analysing the model structure. The folder <code>scripts
</code> contains the current converter.</p> 

## To-Do

<p align="justify">I have to check under which conditions the converter
is working and under which conditions the converter is not working.</p>

<p align="justify">Analysis of the internal model structure to understand
the formatting of different models related to RRDB related models.</p>
 
<p align="justify">I have to catch errors while executing the script.
In the experimental versions I realised this in a rudimentary way.</p>

<p align="justify">Error handling and the output of warnings must be 
improved. Dealing with three similar models harbours the risk of
getting confused with the models.</p>
 
<p align="justify">I need an improvement of the comments and documentation 
in the scripts. And I am also need an improvement of this documentation.</p>

## Test Environment

<p align="justify">I developed and tested the Python scripts using 
the following software development environment:</p>

* Linux Mint 21.3 (Virginia)
* Python 3.10.14
* Numpy  2.1.3
* OpenCV 4.10.0
* PIL 11.0.0
* Torch 2.4.1+cu121
  
## Licenses

<p align="justify">The algorithms with respect to ESRGAN of xinntao are published
under the <i>Apache License</i>. The original scripts and the improved scripts are covered
by this license. I always publish my work under the <i>MIT License</i>.</p> 

## Credits

<p align="justify">My thanks go to the excellent work of Xintao Wang (xinntao). 
The results that can be achieved with his approach are more than good.</p> 

## Reference

[1] https://github.com/xinntao/ESRGAN

[2] https://github.com/xinntao/Real-ESRGAN

[3] https://huggingface.co/

[4] https://openmodeldb.info/

[5] https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth

[6] https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY

[7] https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ

[8] https://ar5iv.labs.arxiv.org/html/1809.00219

[9] https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf

## Donation

<p align="justify">If you like what I present here, or if it helps you,
or if it is useful, you are welcome to donate a small contribution. Or
as you might say: Every TRON counts! Many thanks in advance! :smiley:
</p>

> ###### <p align="left">Crypto Coin Tron</p>

```
TQHgMpVzWkf3Lkd68QS7tcn7S5Y9ayg6Fu
```

> ###### <p align="left">Crypto Coin Dogecoin</p>

```
DQYkNGW8VfCuUbM9Womnp6KiFdtMa4NUkD
```

> ###### <p align="left">Crypto Coin Bitcoin</p>

```
bc1quy5nr5x4xgrtmnknuayysapg8t8tl4jqsgcu2t
```

> ###### <p align="left">Crypto Coin Ethereum</p>

```
0xD238a487F5147AC23c25908FC86129889E38BfB9
```
<p align="left">$${\textnormal{\color{purple}Have a wonderful, beautiful and successful day. I also wish everyone peace on earth.}}$$</p>
