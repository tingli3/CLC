#!/bin/bash
script_dir=$(dirname $0)
cellsize=24
radius=24
#xmin=260350
#ymin=4472061
#xmax=385210
#ymax=4527973
#${script_dir}/lineDensityDiff.sh $shp1 $shp2 $cellsize $radius $xmin $ymin $xmax $ymax

shp1=${script_dir}/streams/DEM18100100.shp
shp2=${script_dir}/streams/NHD18100100.shp

python $script_dir/linedensity_mm.py $shp1 $shp2 $cellsize $radius
#${script_dir}/lineDensityDiff.sh $shp1 $shp2 $cellsize $radius $xmin $ymin $xmax $ymax
