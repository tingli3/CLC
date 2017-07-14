#!/bin/bash
module load gdal2-stack
LineDensity=~/LineDensity2/LineDensity
current_dir=$(pwd)
script_dir=$(dirname $0)
outdir=${script_dir}/result
if [ ! -d $outdir ]; then
 mkdir $outdir
else
 rm -r $outdir
 mkdir $outdir
fi

if [ "$#" -eq 8 ]; then
 LineDataset1=$1
 LineDataset2=$2
 cellsize=$3
 radius=$4
 xmin=$5
 ymin=$6
 xmax=$7
 ymax=$8
 LineDensity1=${outdir}/$(basename ${LineDataset1} .shp)_density.tif
 LineDensity2=${outdir}/$(basename ${LineDataset2} .shp)_density.tif
 diff_density=${outdir}/diff_density.tif
 echo Output Directory $outdir
 echo Compare $LineDataset1 and $LineDataset2 with cellsize=$cellsize and radius=$radius 
 echo Bounding Box "<$xmin,$ymin,$xmax,$ymax>"
 $LineDensity $LineDataset1 $LineDensity1 $cellsize $radius $xmin $ymin $xmax $ymax
 $LineDensity $LineDataset2 $LineDensity2 $cellsize $radius $xmin $ymin $xmax $ymax
 gdal_calc.py -A $LineDensity1 -B $LineDensity2 --outfile=${diff_density} --calc="A-B"
else
 echo "wrong parameter"
fi

