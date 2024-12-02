# ESRGAN CONVERTER [![GitHub - ESRGAN](https://img.shields.io/badge/GitHub-ESRGAN-2ea44f)](https://github.com/xinntao/ESRGAN) [![GitHub  - RealEsrGAN](https://img.shields.io/badge/GitHub_-RealESRGAN-9933ff)](https://github.com/xinntao/Real-ESRGAN)

---

### CONTRIBUTION

<b><p align="justify">If you like what I present here, or if it 
helps you, or if it is useful, you are welcome to [donate](#Donation)
a small contribution. It motivates me a lot and speeds up my work
a much 😏.</p></b>

---

> [!NOTE]
> I always assume that <i>Linux</i> is used as the operating system. If I
> use <i>Linux</i> command line commands in this documentation, I do not
>  explicitly point this out.

## Preface

<p align="justify">I am writing this README section to clarify some
things for myself. Since I am absolutely new to the field of creating
and using AI models, there are some basic things to clarify.</p>

<p align="justify">The approach I have chosen here does not require any
knowledge of the underlying model. The model is used as it is. So far,
the conversion has worked well on this basis.</p>

<p align="justify">As this is a simple transformation from one 
architecture to another, I don't add anything, remove anything 
or change anything in the weight and bias values.</p>

<p align="justify">During the conversions, I make sure that the data
structures are adhered to exactly. For example, I use dict and
OrderedDict if the underlying model would do the same.</p>

## Motivation

<p align="justify">I am always interested in the simplest way to
solve a problem. The fact that I have to use an external Python
module or integrate the related classes in my scripts has made 
me look for a simpler solution. The result of my thoughts can 
be found here.</p>

## Side Note

<p align="justify">The converters presented are not fail-safe.
It may be that a conversion is successfully carried out, but 
that the model does not work. To overcome this problem I wrote
a small script, which can be found in the test_scripts folder.
</p>

## CONVERTERS

<p align="justify">The names of the converters are self-explanatory.</p>

- oldESRGAN_to_newESRGAN_converter.py
- newESRGAN_to_RealESRGAN_converter.py
- RealESRGAN_to_newESRGAN_converter.py

<p align="justify">Usage:</p>

```
oldESRGAN_to_newESRGAN_converter.py <model_name.pth>
newESRGAN_to_RealESRGAN_converter.py <model_name.pth>
RealESRGAN_to_newESRGAN_converter.py <model_name.pth>

```

# TL;DR

## Possible Conversions

<p align="justify">I am currently considering two scenarios for the
converters.</p>

<p align="justify">The directions of the conversions are shown by
the arrows.</p>

+ Scenario I

```
old ESRGAN → new ESRGAN → RealESRGAN
```

+ Scenario II

```
RealESRGAN → new ESRGAN
```

## How the Converter Works

<p align="justify">Each converter simply changes the 
literal representation of each key with respect to the
related model.</p>

<p align="justify">The change of the literal values of
the keys is realised via a conversion table and substring
replacements. Literal string 1 is replaced by literal
string 2. Substring 1 is replaced by substring 2, etc.
In the end, the literal strings of the keys correspond
to the specifications of the model..</p>

## Why Are This Converters Working

<p align="justify">I refer to the new ESRGAN model as 
a reference for all considerations. Therefore, all
explanations refer to the new ESRGAN model. The other
explanations of the other models can be transferred or
derived from them.</p>

<p align="justify">I wrote two test scripts to show
how it works. The test scripts can be found in the 
folder <code>test_scripts</code> in the main directory.
</p>

<p align="justify">The first script creates a model.
Then I am extracting the state dict of the model.
Afterwards I am printing the state dict of this model
in form of key/value as skeleton. Next I am loading
in a second script the available upscaler. Then I am
printing the content (state dict) of this model in
form of key/value as skeleton. I am writing both 
results in a text file and compare both files. Both
files to my own surprise are identical.

```
python3 ESRGAN_RRDB_model.py > test1.txt
python3 ESRGAN_RRDB_model_test.py > test2.txt

diff -s test1.txt test2.txt

results in

Files test1.txt and test2.txt are identical.

```

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

<p align="justify">For the converter to work, the above structure must
be mapped in relation to the key and the shapes of the tensor as shown.</p>

## References

[1] https://github.com/xinntao/ESRGAN

[2] https://github.com/xinntao/Real-ESRGAN

[3] https://pytorch.org/docs/stable/generated/torch.load.html

[4] https://pytorch.org/tutorials/beginner/saving_loading_models.html

[5] https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

[6] https://pytorch.org/docs/stable/generated/torch.nn.Module.html

[7] https://markaicode.com/mastering-pytorch-load_state_dict/

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
<p align="left">$${\textnormal{\color{purple}Have a wonderful, beautiful and successful day. I also wish everyone peace on earth.}}$$</p>
