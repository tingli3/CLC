import os
from osgeo import gdal
from osgeo import ogr
import sys
from osgeo import osr
import numpy
import math
from fishnet import fishnet

codedir=os.path.dirname(os.path.realpath(__file__))
workdir=os.getcwd()
#use work directory or code directory to store outputs
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

### calculate line density and get the difference
inDriver = ogr.GetDriverByName("ESRI Shapefile")
inDataSource1 = inDriver.Open(lineData1, 0)
inLayer1 = inDataSource1.GetLayer(0)
extent1 = inLayer1.GetExtent()
inDataSource1= None
inDataSource2= inDriver.Open(lineData2, 0)
inLayer2 = inDataSource2.GetLayer(0)
extent2 = inLayer2.GetExtent()
inDataSource2=None

xmin=min(extent1[0],extent2[0])-float(cellsize)
xmax=max(extent1[1],extent2[1])+float(cellsize)
ymin=min(extent1[2],extent2[2])-float(cellsize)
ymax=max(extent1[3],extent2[3])+float(cellsize)

os.system('{0}/lineDensityDiff.sh {1} {2} {3} {4} {5} {6} {7} {8}'.format(codedir,lineData1,lineData2,
cellsize,radius,xmin,ymin,xmax,ymax))
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
### reclassify the difference raster ###
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
print 'NSSDA:'+str(NSSDA)
#print numpy.count_nonzero( numpy.logical_or(diffArray>NSSDA,diffArray < -NSSDA))
#print numpy.count_nonzero((diffArray > NSSDA)|(diffArray<-NSSDA))
#print numpy.where(abs(diffArray)>NSSDA)
#print numpy.min(diffArray)
#print numpy.max(diffArray)
#t=float(numpy.count_nonzero((diffArray > NSSDA)|(diffArray<-NSSDA)))/(cols*rows)
#print t
#print diffArray[numpy.where(abs(diffArray)>NSSDA)]
#print diffArray[abs(diffArray)>NSSDA]
#print numpy.count_nonzero(abs(diffArray)>NSSDA)
#print diffArray

print "Reclassify..."
diffClassArray=numpy.full((rows,cols),2)
diffClassArray[diffArray<-NSSDA]=1
diffClassArray[diffArray>NSSDA]=3
lineDensity1=os.path.splitext(os.path.basename(lineData1))[0]+'_density.tif'
lineDensity2=os.path.splitext(os.path.basename(lineData2))[0]+'_density.tif'

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

diffClassArray.astype(numpy.int16)
diffClassBand.WriteArray(diffClassArray)
diffClassBand.FlushCache()
diffClass=None

diffArray=None
densityArray1=None
densityArray2=None
#os.system('gdalinfo -mm '+ diffClassRaster)
#os.system('gdal_polygonize.py {0} -f "ESRI Shapefile" {1}/diff.shp "diff.shp" "clc_code"'.format(diffClassRaster,outdir))


### identity line dataset with the reclassified diff raster of line density ###
print 'Identity...'
inDriver = ogr.GetDriverByName("ESRI Shapefile")
inDataSource1 = inDriver.Open(lineData1, 0)
inLayer1 = inDataSource1.GetLayer(0)
inDataSource2= inDriver.Open(lineData2, 0)
inLayer2 = inDataSource2.GetLayer(0)
#memDriver=ogr.GetDriverByName('MEMORY')
#memSource=memDriver.CreateDataSource('memData')
#tmp=memDriver.Open('memData',1)
#diff_mem=memSource.CopyLayer(diffLayer)

srs=inLayer1.GetSpatialRef()
output1=outdir+'/{0}_identity.shp'.format(os.path.splitext(os.path.basename(lineData1))[0])
outIdentity1=inDriver.CreateDataSource(output1)
outLayer1=outIdentity1.CreateLayer('outIdentity1.shp',srs,ogr.wkbLineString)
#outLayer1=outIdentity1.CreateLayer('outIdentity1.shp',srs,inLayer1.GetGeomType())

output2=outdir+'/{0}_identity.shp'.format(os.path.splitext(os.path.basename(lineData2))[0])
srs=inLayer2.GetSpatialRef()
outIdentity2=inDriver.CreateDataSource(output2)
outLayer2=outIdentity2.CreateLayer('outIdentity2.shp',srs,ogr.wkbLineString)
#outLayer2=outIdentity2.CreateLayer('outIdentity2.shp',srs,inLayer2.GetGeomType())

inLayerDefn1 = inLayer1.GetLayerDefn()
for i in range(0,inLayerDefn1.GetFieldCount()):
	fieldDefn = inLayerDefn1.GetFieldDefn(i)
	outLayer1.CreateField(fieldDefn)
outLayer1.CreateField(ogr.FieldDefn("clc_code",ogr.OFTInteger))
outLayerDefn1=outLayer1.GetLayerDefn()

inLayerDefn2 = inLayer2.GetLayerDefn()
for i in range(0,inLayerDefn2.GetFieldCount()):
	fieldDefn = inLayerDefn2.GetFieldDefn(i)
	outLayer2.CreateField(fieldDefn)
outLayer2.CreateField(ogr.FieldDefn("clc_code",ogr.OFTInteger))
outLayerDefn2=outLayer2.GetLayerDefn()


step=float(cellsize)/4

for i_feature in range(0, inLayer1.GetFeatureCount()):
	inFeature = inLayer1.GetFeature(i_feature)
	geom = inFeature.GetGeometryRef()
	for i_geometry in range (0, max(1,geom.GetGeometryCount())):
		if i_geometry == 0 and geom.GetGeometryCount() == 0:
			g=geom
		else:
			g=geom.GetGeometryRef(i_geometry)
		pts=[]
		for i_point in range(0,g.GetPointCount()-1):
			start=g.GetPoint_2D(i_point)
			pts+=[start]
			end=g.GetPoint_2D(i_point+1)
			L=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
			if L == 0: continue
			u=[(end[0]-start[0])*step/L,(end[1]-start[1])*step/L]
			cx=start[0]+u[0]
			cy=start[1]+u[1]
			i_start=int((start[1]-originY)/pixelHeight)
			j_start=int((start[0]-originX)/pixelWidth)
			while( (cx-start[0])*(cx-end[0])<0 or (cy-start[1])*(cy-end[1])<0 ):
				i=int((cy-originY)/pixelHeight)
				j=int((cx-originX)/pixelWidth)
				if diffClassArray[i,j] != diffClassArray[i_start,j_start]:
					pts+=[(cx,cy)]
					clc=diffClassArray[i_start,j_start]
					#create a new line
					outFeature=ogr.Feature(outLayerDefn1)
					for i_field in range(0,outLayerDefn1.GetFieldCount()-1):
						#outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
						outFeature.SetField(i_field,inFeature.GetField(i_field))
					outFeature.SetField("clc_code",clc)
					newLine=ogr.Geometry(ogr.wkbLineString)
					for i_pt in range(0,len(pts)):
						newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
					outFeature.SetGeometry(newLine)
					outLayer1.CreateFeature(outFeature)
					outFeature=None
					pts=[]
					pts+=[(cx,cy)]
					start=(cx,cy)
					i_start=int((start[1]-originY)/pixelHeight)
					j_start=int((start[0]-originX)/pixelWidth)
				cx=cx+u[0]
				cy=cy+u[1]
			i_end=int((end[1]-originY)/pixelHeight)
			j_end=int((end[0]-originX)/pixelWidth)
			if diffClassArray[i_end,j_end] != diffClassArray[i_start,j_start]:
				pts+=[end]
				clc=diffClassArray[i_start,j_start]
				outFeature=ogr.Feature(outLayerDefn1)
				for i_field in range(0,outLayerDefn1.GetFieldCount()-1):
					#outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
					outFeature.SetField(i_field,inFeature.GetField(i_field))
				outFeature.SetField("clc_code",clc)
				newLine=ogr.Geometry(ogr.wkbLineString)
				for i_pt in range(0,len(pts)):
					newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
				outFeature.SetGeometry(newLine)
				outLayer1.CreateFeature(outFeature)
				outFeature=None
				pts=[]
		pts += [g.GetPoint_2D(g.GetPointCount()-1)]
		if len(pts) > 1:
			i_start=int((start[1]-originY)/pixelHeight)
			j_start=int((start[0]-originX)/pixelWidth)
			clc=diffClassArray[i_start,j_start]
			outFeature=ogr.Feature(outLayerDefn1)
			for i_field in range(0,outLayerDefn1.GetFieldCount()-1):
				#outFeature.SetField(outLayerDefn1.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
				outFeature.SetField(i_field,inFeature.GetField(i_field))
			outFeature.SetField("clc_code",clc)
			newLine=ogr.Geometry(ogr.wkbLineString)
			for i_pt in range(0,len(pts)):
				newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
			outFeature.SetGeometry(newLine)
			outLayer1.CreateFeature(outFeature)
			outFeature=None


for i_feature in range(0, inLayer2.GetFeatureCount()):
	inFeature = inLayer2.GetFeature(i_feature)
	geom = inFeature.GetGeometryRef()
	for i_geometry in range (0, max(1,geom.GetGeometryCount())):
		if i_geometry == 0 and geom.GetGeometryCount() == 0:
			g=geom
		else:
			g=geom.GetGeometryRef(i_geometry)
		pts=[]
		for i_point in range(0,g.GetPointCount()-1):
			start=g.GetPoint_2D(i_point)
			pts+=[start]
			end=g.GetPoint_2D(i_point+1)
			L=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
			if L==0: continue
			u=[(end[0]-start[0])*step/L,(end[1]-start[1])*step/L]
			cx=start[0]+u[0]
			cy=start[1]+u[1]
			i_start=int((start[1]-originY)/pixelHeight)
			j_start=int((start[0]-originX)/pixelWidth)
			while( (cx-start[0])*(cx-end[0])<0 or (cy-start[1])*(cy-end[1])<0 ):
				i=int((cy-originY)/pixelHeight)
				j=int((cx-originX)/pixelWidth)
				if diffClassArray[i,j] != diffClassArray[i_start,j_start]:
					pts+=[(cx,cy)]
					clc=diffClassArray[i_start,j_start]
					#create a new line
					outFeature=ogr.Feature(outLayerDefn2)
					for i_field in range(0,outLayerDefn2.GetFieldCount()-1):
						#outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
						outFeature.SetField(i_field,inFeature.GetField(i_field))
					outFeature.SetField("clc_code",clc)
					newLine=ogr.Geometry(ogr.wkbLineString)
					for i_pt in range(0,len(pts)):
						newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
					outFeature.SetGeometry(newLine)
					outLayer2.CreateFeature(outFeature)
					outFeature=None
					pts=[]
					pts+=[(cx,cy)]
					start=(cx,cy)
					i_start=int((start[1]-originY)/pixelHeight)
					j_start=int((start[0]-originX)/pixelWidth)
				cx=cx+u[0]
				cy=cy+u[1]
			i_end=int((end[1]-originY)/pixelHeight)
			j_end=int((end[0]-originX)/pixelWidth)
			if diffClassArray[i_end,j_end] != diffClassArray[i_start,j_start]:
				pts+=[end]
				clc=diffClassArray[i_start,j_start]
				outFeature=ogr.Feature(outLayerDefn2)
				for i_field in range(0,outLayerDefn2.GetFieldCount()-1):
					#outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
					outFeature.SetField(i_field,inFeature.GetField(i_field))
				outFeature.SetField("clc_code",clc)
				newLine=ogr.Geometry(ogr.wkbLineString)
				for i_pt in range(0,len(pts)):
					newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
				outFeature.SetGeometry(newLine)
				outLayer2.CreateFeature(outFeature)
				outFeature=None
				pts=[]
		pts += [g.GetPoint_2D(g.GetPointCount()-1)]
		if len(pts) > 1:
			i_start=int((start[1]-originY)/pixelHeight)
			j_start=int((start[0]-originX)/pixelWidth)
			clc=diffClassArray[i_start,j_start]
			outFeature=ogr.Feature(outLayerDefn2)
			for i_field in range(0,outLayerDefn2.GetFieldCount()-1):
				#outFeature.SetField(outLayerDefn2.GetFieldDefn(i).GetNameRef(),inFeature.GetField(i_field))
				outFeature.SetField(i_field,inFeature.GetField(i_field))
			outFeature.SetField("clc_code",clc)
			newLine=ogr.Geometry(ogr.wkbLineString)
			for i_pt in range(0,len(pts)):
				newLine.AddPoint(pts[i_pt][0],pts[i_pt][1])
			outFeature.SetGeometry(newLine)
			outLayer2.CreateFeature(outFeature)
			outFeature=None


### calculate total length of different CLC value ###
print "Calculate length of different CLC value..."
#s1=dict(clc1=0,clc2=0,clc3=0,clc4=0)
#s2=dict(clc1=0,clc2=0,clc3=0,clc4=0)
outLayer1.ResetReading()
outLayer2.ResetReading()
s1={1:0,2:0,3:0,4:0}
s2={1:0,2:0,3:0,4:0}
for feature in outLayer1:
	#temp='clc'+str(feature.GetField('clc_code'))
	temp=feature.GetField('clc_code')
	geom=feature.GetGeometryRef()
	s1[temp]+=geom.Length()

for feature in outLayer2:
	#temp='clc'+str(feature.GetField('clc_code'))
	temp=feature.GetField('clc_code')
	geom=feature.GetGeometryRef()
	s2[temp]+=geom.Length()
	#print geom.Length()
       #print 'clc: {0}   ID: {1}'.format(feature.GetField('clc_code'),feature.GetField(0))
inDataSource1=None
inDataSource2=None

print 'clc code and corresponding total length for {0}, clc code : total length'.format(lineData1)
print s1
match1=s1[1]+s1[2]
print 'total length of matching features in {0}: {1}'.format(lineData1,match1)
mis1=s1[3]+s1[4]
print 'total length of mismatching features in {0}: {1}'.format(lineData1,mis1)
print 'clc code and corresponding total length for {0}, clc code : total length'.format(lineData2)
print s2
match2=s2[2]+s2[3]
print 'total length of matching features in {0}: {1}'.format(lineData2,match2)
mis2=s2[1]+s2[4]
print 'total length of mismatching features in {0}: {1}'.format(lineData2,mis2)
print 'total CLC = {}'.format((match1+match2)/(match1+mis1+match2+mis2))

### create fishnet ###
fishnetData=outdir+'/fishnet.shp'
nMax=240
nMin=200
gridsize = (int)(math.sqrt((abs(xmax-xmin) * abs(ymax-ymin))/nMin))
gridcount=int(math.ceil(abs(xmax-xmin)/gridsize)*math.ceil(abs(ymax-ymin)/gridsize))

if gridcount > nMax or gridcount < nMin:
	gridsize=math.sqrt((abs(xmax-xmin)*abs(ymax-ymin))/nMin)
	gridcount=int(math.ceil(abs(xmax-xmin)/gridsize)*math.ceil(abs(ymax-ymin)/gridsize))

print 'Creating fishnet using gridsize = {0} gridcount = {1}'.format(gridsize,gridcount)
fishnet(fishnetData,srs,xmin,xmax,ymin,ymax,gridsize,gridsize)
### calculate CLC value in each fishnet grid ###
print 'Identify lines with fishnet grid...'


srs=outLayer1.GetSpatialRef()
outfish1=outdir+'/{0}_gridded.shp'.format(os.path.splitext(os.path.basename(lineData1))[0])
fishIdentity1=inDriver.CreateDataSource(outfish1)
gridLayer1=fishIdentity1.CreateLayer('fishIdentity1.shp',srs,ogr.wkbLineString)
outfish2=outdir+'/{0}_gridded.shp'.format(os.path.splitext(os.path.basename(lineData2))[0])
srs=outLayer2.GetSpatialRef()
fishIdentity2=inDriver.CreateDataSource(outfish2)
gridLayer2=fishIdentity2.CreateLayer('fishIdentity2.shp',srs,ogr.wkbLineString)

fishDataSource=inDriver.Open(fishnetData,0)
fishLayer=fishDataSource.GetLayer()

outLayer1.Identity(fishLayer,gridLayer1,['SKIP_FAILURES=YES','PROMOTE_TO_MULTI=NO','KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])
outLayer2.Identity(fishLayer,gridLayer2,['SKIP_FAILURES=YES','PROMOTE_TO_MULTI=NO','KEEP_LOWER_DIMENSION_GEOMETRIES=NO'])

outIdentity2=None
outIdentity1=None

print 'Calculating CLC value in each grid...'
srs=fishLayer.GetSpatialRef()
outgrid=outdir+'/grid_CLC.shp'
outGridSource=inDriver.CreateDataSource(outgrid)
finalGrid=outGridSource.CreateLayer('grid_CLC.shp',srs,ogr.wkbPolygon)
finalGrid.CreateField(ogr.FieldDefn('fishID',ogr.OFTInteger))
finalGrid.CreateField(ogr.FieldDefn('bm_clc1',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc2',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc3',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_clc4',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc1',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc2',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc3',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_clc4',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_match',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('bm_omi',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_match',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('tg_commi',ogr.OFTReal))
finalGrid.CreateField(ogr.FieldDefn('full_clc',ogr.OFTReal))
finalGridDefn=finalGrid.GetLayerDefn()
outdict={}

fishLayer.ResetReading()
gridLayer1.ResetReading()
gridLayer2.ResetReading()
for feature in gridLayer1:
	fishid=feature.GetField('fishID')
	if fishid not in outdict:
		outdict[fishid]=[0.0 for x in range(0,8)]
	clc=feature.GetField('clc_code')
	geom=feature.GetGeometryRef()
	outdict[fishid][clc-1] += geom.Length()

for feature in gridLayer2:
	fishid=feature.GetField('fishID')
	if fishid not in outdict:
		outdict[fishid]=[0.0 for x in range(0,8)]
	clc=feature.GetField('clc_code')
	geom=feature.GetGeometryRef()
	outdict[fishid][clc+3] += geom.Length()

for feature in fishLayer:
	fishid=feature.GetField('fishID')
	if fishid not in outdict: 
		continue
	else:
		geom=feature.GetGeometryRef()
		outFeature=ogr.Feature(finalGridDefn)
		outFeature.SetGeometry(geom)
		outFeature.SetField('fishID',fishid)
		temp=outdict[fishid]
		for i in range(0,8):
			outFeature.SetField(i+1,temp[i])
		match1=temp[0]+temp[1]
		mis1=temp[2]+temp[3]
		match2=temp[5]+temp[6]
		mis2=temp[4]+temp[7]
		outFeature.SetField('bm_match',match1)
		outFeature.SetField('bm_omi',mis1)
		outFeature.SetField('tg_match',match2)
		outFeature.SetField('tg_commi',mis2)
		fullCLC=(match1+match2)/(match1+mis1+match2+mis2)
		outFeature.SetField('full_clc',fullCLC)
		finalGrid.CreateFeature(outFeature)
		outFeature=None

fishIdentity1=None
fishIdentity2=None
fishDataSource=None
outGridSource=None
