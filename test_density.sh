#!/bin/bash
script_dir=$(dirname $0)
#shp1=${script_dir}/streams/mac_stream_1000.shp
#shp2=${script_dir}/streams/mac_stream_10000.shp
cellsize=10
radius=100
xmin=259809
ymin=4471270
xmax=385474
ymax=4528723
#${script_dir}/lineDensityDiff.sh $shp1 $shp2 $cellsize $radius $xmin $ymin $xmax $ymax

shp=${script_dir}/streams/mac_stream_100.shp

#${script_dir}/lineDensity.sh $shp ${script_dir}/result/test1.tif $cellsize $radius
${script_dir}/lineDensity.sh $shp ${script_dir}/result/test1m.tif $cellsize $radius $xmin $ymin $xmax $ymax
