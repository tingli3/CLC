import os
from osgeo import gdal
from osgeo import ogr
import sys
from osgeo import osr
import numpy
import math

codedir=os.path.dirname(os.path.realpath(__file__))
workdir=os.getcwd()
#print workdir
outdir=codedir+'/result'
if len(sys.argv) < 5:
    print('usage: {} <shapefile1> <shapefile2> <cellsize> <radius>'.format(sys.argv[0]))
    sys.exit(-1)

lineData1=sys.argv[1]
lineData2=sys.argv[2]
cellsize=sys.argv[3]
radius=sys.argv[4]

#source_ds = ogr.Open(vector_fn)
#source_layer = source_ds.GetLayer()
#x_min, x_max, y_min, y_max = source_layer.GetExtent()


inDriver = ogr.GetDriverByName("ESRI Shapefile")
inDataSource1 = inDriver.Open(lineData1, 0)
inLayer1 = inDataSource1.GetLayer(0)
extent1 = inLayer1.GetExtent()
inDataSource1= None
inDataSource2= inDriver.Open(lineData2, 0)
inLayer2 = inDataSource2.GetLayer(0)
extent2 = inLayer2.GetExtent()
inDataSource2=None
#print extent1,extent2

xmin=min(extent1[0],extent2[0])
xmax=max(extent1[1],extent2[1])
ymin=min(extent1[2],extent2[2])
ymax=max(extent1[3],extent2[3])
#print '<'+str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax)+'>'
#print 'Use Bounding Box: <{0},{1},{2},{3}>'.format(xmin,ymin,xmax,ymax)

os.system('{0}/lineDensityDiff.sh {1} {2} {3} {4} {5} {6} {7} {8}'.format(codedir,lineData1,lineData2,
cellsize,radius,xmin,ymin,xmax,ymax))
#print '{0}/diff_density.tif'.format(outdir)
diffRaster=gdal.Open('{0}/diff_density.tif'.format(outdir))
diffBand=diffRaster.GetRasterBand(1)
geotransform = diffRaster.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
cols = diffRaster.RasterXSize
rows = diffRaster.RasterYSize
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]
outRasterSRS = osr.SpatialReference()
outRasterSRS.ImportFromWkt(diffRaster.GetProjectionRef())
diffArray=diffBand.ReadAsArray().astype(numpy.float)
diffRaster=None

diffClassRaster='{0}/diff_class.tif'.format(outdir)
driver = gdal.GetDriverByName('GTiff')
diffClass = driver.Create(diffClassRaster, cols, rows, 1, gdal.GDT_Int16)
diffClass.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
diffClassBand = diffClass.GetRasterBand(1)
diffClass.SetProjection(outRasterSRS.ExportToWkt())

sqDiff=numpy.square(diffArray)
mean=numpy.mean(sqDiff)
RMSE=math.sqrt(mean)
NSSDA=1.96*RMSE
sqDiff= None
#print rows,cols
#print diffArray.shape
print 'NSSDA:'+str(NSSDA)
#print numpy.count_nonzero( numpy.logical_or(diffArray>NSSDA,diffArray < -NSSDA))
#print numpy.count_nonzero((diffArray > NSSDA)|(diffArray<-NSSDA))
#print numpy.where(abs(diffArray)>NSSDA)
#print numpy.min(diffArray)
#print numpy.max(diffArray)
#t=float(numpy.count_nonzero((diffArray > NSSDA)|(diffArray<-NSSDA)))/(cols*rows)
#print t
#print cols,rows
#
#print diffArray[numpy.where(abs(diffArray)>NSSDA)]
#print diffArray[abs(diffArray)>NSSDA]
#print numpy.count_nonzero(abs(diffArray)>NSSDA)

#print diffArray
diffClassArray=numpy.full((rows,cols),2)
diffClassArray[diffArray<-NSSDA]=1
diffClassArray[diffArray>NSSDA]=3
lineDensity1=os.path.splitext(os.path.basename(lineData1))[0]+'_density.tif'
lineDensity2=os.path.splitext(os.path.basename(lineData2))[0]+'_density.tif'
#print lineDensity1
#print lineDensity2

densityRaster1=gdal.Open('{0}/{1}'.format(outdir,lineDensity1))
densityRaster2=gdal.Open('{0}/{1}'.format(outdir,lineDensity2))
densityBand1=densityRaster1.GetRasterBand(1)
densityBand2=densityRaster2.GetRasterBand(1)
densityArray1=densityBand1.ReadAsArray().astype(numpy.float)
densityArray2=densityBand2.ReadAsArray().astype(numpy.float)
densityRaster1=None
densityRaster2=None

diffClassArray[densityArray1 < 0.000000001]=4
diffClassArray[densityArray2 < 0.000000001]=4



#diffClassArray[]
diffClassArray.astype(numpy.int16)
diffClassBand.WriteArray(diffClassArray)
diffClassBand.FlushCache()
diffClass=None
#os.system('gdalinfo -mm '+ diffClassRaster)
os.system('gdal_polygonize.py {0} -f "ESRI Shapefile" {1}/diff.shp "diff.shp" "clc_code"'.format(diffClassRaster,outdir))


inDriver = ogr.GetDriverByName("ESRI Shapefile")
inDataSource1 = inDriver.Open(lineData1, 0)
inLayer1 = inDataSource1.GetLayer(0)
inDataSource2= inDriver.Open(lineData2, 0)
inLayer2 = inDataSource2.GetLayer(0)
diffPolyData=outdir+'/diff.shp'
diffPolyDataSource=inDriver.Open(diffPolyData,0)
diffLayer=diffPolyDataSource.GetLayer(0)


#memDriver=ogr.GetDriverByName('MEMORY')
#memSource=memDriver.CreateDataSource('memData')
#tmp=memDriver.Open('memData',1)
#diff_mem=memSource.CopyLayer(diffLayer)




srs=inLayer1.GetSpatialRef()
output1=outdir+'/outIdentity1.shp'
outIdentity1=inDriver.CreateDataSource(output1)
#outLayer1=outIdentity1.CreateLayer('outIdentity1.shp',srs,ogr.wkbLineString)
outLayer1=outIdentity1.CreateLayer('outIdentity1.shp',srs,inLayer1.GetGeomType())

output2=outdir+'/outIdentity2.shp'
srs=inLayer2.GetSpatialRef()
outIdentity2=inDriver.CreateDataSource(output2)
#outLayer2=outIdentity2.CreateLayer('outIdentity2.shp',srs,ogr.wkbLineString)
outLayer2=outIdentity2.CreateLayer('outIdentity2.shp',srs,inLayer2.GetGeomType())

diffLayer.SetAttributeFilter("clc_code != 4")
inLayer1.Identity(diffLayer, outLayer1,['SKIP_FAILURES=YES','PROMOTE_TO_MULTI=NO','KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])
inLayer2.Identity(diffLayer, outLayer2,['SKIP_FAILURES=YES','PROMOTE_TO_MULTI=NO','KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])

#inLayer1.Identity(diffLayer, outLayer1,['SKIP_FAILURES=YES','PROMOTE_TO_MULTI=NO','KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])


#outLayer1.ResetReading()


for feature in outLayer1:
	if feature.GetField('clc_code') == None:
		feature.SetField('clc_code',4) 
		#print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
		outLayer1.SetFeature(feature)
outLayer1.ResetReading()


for feature in outLayer2:
	if feature.GetField('clc_code') == None:
		feature.SetField('clc_code',4) 
		#print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
		outLayer2.SetFeature(feature)
outLayer2.ResetReading()


s1=dict(clc1=0,clc2=0,clc3=0,clc4=0)
s2=dict(clc1=0,clc2=0,clc3=0,clc4=0)
for feature in outLayer1:
	temp='clc'+str(feature.GetField('clc_code'))
	geom=feature.GetGeometryRef()
	s1[temp]+=geom.Length()

for feature in outLayer2:
	temp='clc'+str(feature.GetField('clc_code'))
	geom=feature.GetGeometryRef()
	s2[temp]+=geom.Length()
	#print geom.Length()
       #print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
outLayer2.ResetReading()
#print outLayer2.GetFeatureCount()
diffPolyDataSource=None
inDataSource1=None
inDataSource2=None
outIdentity2=None
outIdentity1=None

#outIdentity1=inDriver.Open(output1,0)
#outLayer1=outIdentity1.GetLayer(0)

#for feature in outLayer1:
#       print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
#outLayer1.ResetReading()
print s1
print s2
