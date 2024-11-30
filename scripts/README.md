# Content

## RRDBNet_arch.py

Python module with the main classes for RRDB. Needs to be be in the same directory as converter_RRDB_models.py and experimental_converter_RRDB_models.py.

---

## basic_converter_RRDB_models.py

Converter for old ESRGAN to new ESRGAN models. Based direct on the conversion application of xinntao.

## experimental_converter_RRDB_models.py

Experimenta converter for old ESRGAN to new ESRGAN models. Something else based on the methodology of the conversion application of xinntao.

---

## realesrgan_converter_RRDB_models.py

Converter for RealESRGAN to new ESRGAN models. Something else based on the methodology of the conversion application of xinntao.

---

## realesrgan_converter.py

Converter for new ESRGAN models to RealESRGAN models. Used (still) 
the naming of xinntao. Otherwise complete in-house development.
Module RRDBNet_arch.py no longer required. Standalone working.

---

## RealESRGAN_test.py

Upscaler for RealESRGAN model.

Usage:

````
python3 RealESRGAN_test.py "image.jpg" "model.pth" "scale" "device"
````
