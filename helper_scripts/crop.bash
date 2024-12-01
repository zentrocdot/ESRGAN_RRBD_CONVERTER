#!usr/bin/bash

#FN="shedevil.jpg"
#FN="esrgan.jpg"
FN="realesrgan.jpg"

factor=4

x=$((512*factor))
y=$((512*factro))

x0=$((115*factor))
y0=$((115*factor))

x1=$((205*factor))
y1=$((85*factor))

magick "${FN}" -crop ${x0}x${y0}+${x1}+${y1} cropped.jpg

exit 0
