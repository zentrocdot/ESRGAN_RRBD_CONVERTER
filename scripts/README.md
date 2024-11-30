# Content

## RRDBNet_arch.py

Python module with the main classes for RRDB. Needs to be be in the same directory as converter_RRDB_models.py and experimental_converter_RRDB_models.py.

---

## converter_RRDB_models.py

Converter for old ESRGAN to new ESRGAN models. Based direct on the conversion application of xinntao.

## experimental_converter_RRDB_models.py

Experimenta converter for old ESRGAN to new ESRGAN models. Something else based on the methodology of the conversion application of xinntao.

---

## RealESRGAN_test.py

Upscaler for RealESRGAN model.

Usage:

````
python3 RealESRGAN_test.py "image.jpg" "model.pth" "scale" "device"
````
