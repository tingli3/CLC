#!/bin/bash
module load gdal2-stack
LineDensity=~/LineDensity2/LineDensity
current_dir=$(pwd)
script_dir=$(dirname $0)

if [ "$#" -eq 4 ]; then
 LineDataset=$1
 Output=$2
 cellsize=$3
 radius=$4
 echo "using $LineDensity"
 echo "input=$LineDataset output=$Output cellsize=$cellsize radius=$radius"
 $LineDensity $LineDataset $Output $cellsize $radius
elif [ "$#" -eq 8 ]; then
 LineDataset=$1
 Output=$2
 cellsize=$3
 radius=$4
 xmin=$5
 ymin=$6
 xmax=$7
 ymax=$8
 echo "using $LineDensity"
 echo "input=$LineDataset output=$Output cellsize=$cellsize radius=$radius bounding box: <$xmin, $ymin, $xmax, $ymax>"
 $LineDensity $LineDataset $Output $cellsize $radius $xmin $ymin $xmax $ymax
else
 echo "wrong parameter"
fi

