#!/usr/bin/bash

factor=1

x1=$((205*factor))
y1=$((85*factor))

w=$((115*factor))
h=$((115*factor))

x2=$((x1 + w))
y2=$((y1 + h))

magick "shedevil.jpg" -stroke purple -strokewidth 2.5 -fill "rgba( 255, 215, 0 , 0.0 )" -draw "rectangle $x1,$y1 320,200" out_fill.jpg

exit 0
