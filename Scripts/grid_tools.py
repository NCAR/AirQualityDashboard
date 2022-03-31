# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
# Copyright UCAR (c) 2020
# University Corporation for Atmospheric Research(UCAR)
# National Center for Atmospheric Research(NCAR)
# Research Applications Laboratory(RAL)
# P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
#
# Name:        module1
# Purpose:
# Author:      $ Kevin Sampson(ksampson)
# Created:     2022
# Licence:     <your licence>
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

'''
2022/03/30
    This script is designed to store data and functions in support of the Air
    Quality dashboard.
'''

# --- Import Modules --- #

# Import Python Core Modules
import time
import os
import sys
from distutils.version import LooseVersion

# Import Additional Modules
import netCDF4
import numpy
import osgeo
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdalconst
from osgeo.gdal_array import *                # Assists in using BandWriteArray, BandReadAsArray, and CopyDatasetInfo

# Library options
sys.dont_write_bytecode = True
gdal.UseExceptions()                                                            # this allows GDAL to throw Python Exceptions
gdal.PushErrorHandler('CPLQuietErrorHandler')

# --- End Import Modules --- #

# --- Global Variables --- #

# Air Quality grid information
DX = 12000.0
DY = -12000.0
nrows = 265
ncols = 442
x00 = -2508000.256
y00 = 1464001.179
grid_proj4 = '+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs'

# Obtain the path of this file, so other files may be found relative to it
file_path = os.path.abspath(os.path.dirname(__file__))
OutDir = os.path.abspath(os.path.dirname(__file__))

# A raster may be used to describe the model grid, rather than specifying with globals above
fromRaster = False
raster_path = os.path.join(file_path, r'..\Data\GIS\Tiff\pm25_daily.tif')

# Specify polygon file for calculating boundaries
use_poygons = True
polygons = os.path.join(file_path, r'..\Data\GIS\Boundaries\Geojsons\US_States.geojson')
fieldname = 'NAME'

# Use pre-calculated spatial weight file?
read_weights = False
regridweightnc = os.path.join(file_path, r'..\Data\weights\US_States_AQgrid.nc')

# Other options
check_geometry = True                                                           # Check geometry to trap any .Area issues
threshold = 1e10                                                                # Number of square meters above which a polygon gets split up
splits = 2                                                                      # Number of chunks to divide the large polygon into
# --- End Global Variables --- #

# --- Classes --- #
class Gridder_Layer(object):
    '''Class with which to create the grid intersecting grid cells based on a feature
    geometry envelope. Provide grid information to initiate the class, and use getgrid()
    to generate a grid mesh and index information about the intersecting cells.

    Note:  The i,j index begins with (1,1) in the Lower Left corner.'''
    def __init__(self, DX, DY, x00, y00, nrows, ncols, proj4):

        self.DX = DX
        self.DY = DY
        self.x00 = x00
        self.y00 = y00
        self.nrows = nrows
        self.ncols = ncols
        self.proj4 = proj4

        # create the spatial reference for the input, based on provided proj.4
        proj = osr.SpatialReference()
        proj.ImportFromProj4(proj4)

        # Added 11/19/2020 to allow for GDAL 3.0 changes to the order of coordinates in transform
        if int(osgeo.__version__[0]) >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            proj.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.proj = proj

    def GeoTransform(self):
        '''
        Return the affine transformation for this grid. Assumes a 0 rotation grid.
        (top left x, w-e resolution, 0=North up, top left y, 0 = North up, n-s pixel resolution (negative value))
        '''
        return (self.x00, self.DX, 0, self.y00, 0, self.DY)

    def grid_extent(self):
        '''
        Return the grid bounding extent [xMin, yMin, xMax, yMax]
        '''
        xMax = self.x00 + (float(self.ncols)*self.DX)
        yMin = self.y00 + (float(self.nrows)*self.DY)
        return [self.x00, yMin, xMax, self.y00]

    def grid_envelope(self):
        '''
        Return the grid bounding envelope [xMin, xMax, yMin, yMax]
        '''
        # Calculate the number of grid cells necessary
        xmin, ymin, xmax, ymax = self.grid_extent()
        envelope = [xmin, xmax, ymin, ymax]
        return envelope

    def boundarySHP(self, outputFile, DriverName='ESRI Shapefile'):
        '''Build a single-feature rectangular polygon that represents the boundary
        of the WRF/WRF-Hydro domain. '''

        # Now convert it to a vector file with OGR
        tic1 = time.time()
        drv = ogr.GetDriverByName(DriverName)
        if drv is None:
            print('  {0} driver not available.'.format(DriverName))
        else:
            print('  {0} driver is available.'.format(DriverName))
            datasource = drv.CreateDataSource(outputFile)
        if datasource is None:
            print('  Creation of output file failed.\n')
            raise SystemExit

        # Create output polygon vector file
        layer = datasource.CreateLayer('grid_boundary', self.proj, geom_type=ogr.wkbPolygon)
        if layer is None:
            print('  Layer creation failed.\n')
            raise SystemExit
        LayerDef = layer.GetLayerDefn()                                             # Fetch the schema information for this layer

        # Create polygon object that is fully inside the outer edge of the domain
        [xMin, yMin, xMax, yMax] = self.grid_extent()
        ring = ogr.Geometry(type=ogr.wkbLinearRing)
        ring.AddPoint(xMin, yMax)
        ring.AddPoint(xMax, yMax)
        ring.AddPoint(xMax, yMin)
        ring.AddPoint(xMin, yMin)
        ring.AddPoint(xMin, yMax)                                     #close ring
        geometry = ogr.Geometry(type=ogr.wkbPolygon)
        geometry.AssignSpatialReference(self.proj)
        geometry.AddGeometry(ring)

        # Create the feature
        feature = ogr.Feature(LayerDef)                                     # Create a new feature (attribute and geometry)
        feature.SetGeometry(geometry)                                      # Make a feature from geometry object
        layer.CreateFeature(feature)
        print('  Done producing output vector polygon shapefile in {0: 3.3f} seconds'.format(time.time()-tic1))
        datasource = ring = feature = layer = None
        return geometry

    def getgrid(self, envelope, layer):
        """Gridder.getgrid() takes as input an OGR geometry envelope, and will
        compute the grid polygons that intersect the evelope, returning a list
        of grid cell polygons along with other attribute information."""

        # Calculate the number of grid cells necessary
        xmin, xmax, ymin, ymax = envelope

        # Find the i and j indices
        i0 = int((xmin-self.x00)/self.DX // 1)                                  # Floor the value
        j0 = int(abs((ymax-self.y00)/self.DY) // 1)                             # Floor the absolute value
        i1 = int((xmax-self.x00)/self.DX // 1)                                  # Floor the value
        j1 = int(abs((ymin-self.y00)/self.DY) // 1)                             # Floor the absolute value

        # Create a new field on a layer. Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('i_index', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('j_index', ogr.OFTInteger))
        LayerDef = layer.GetLayerDefn()                                         # Fetch the schema information for this layer

        # Build OGR polygon objects for each grid cell in the intersecting envelope
        for x in range(i0, i1+1):
            if x < 0 or x > self.ncols:
                continue
            for y in reversed(range(j0, j1+1)):
                if y < 0 or y > self.nrows:
                    continue
                id1 = (self.nrows*(x+1))-y                                      # This should give the ID of the cell from the lower left corner (1,1)

                # Calculating each grid cell polygon's coordinates
                x0 = self.x00 + (self.DX*x)
                x1 = x0 + self.DX
                y1 = self.y00 - (abs(self.DY)*y)
                y0 = y1 - abs(self.DY)

                # Create ORG geometry polygon object using a ring
                myRing = ogr.Geometry(type=ogr.wkbLinearRing)
                myRing.AddPoint(x0, y1)
                myRing.AddPoint(x1, y1)
                myRing.AddPoint(x1, y0)
                myRing.AddPoint(x0, y0)
                myRing.AddPoint(x0, y1)
                geometry = ogr.Geometry(type=ogr.wkbPolygon)
                geometry.AddGeometry(myRing)

                # Create the feature
                feature = ogr.Feature(LayerDef)                                     # Create a new feature (attribute and geometry)
                feature.SetField('id', id1)
                feature.SetField('cellsize', geometry.Area())
                feature.SetField('i_index', x+1)
                feature.SetField('j_index', self.nrows-y)
                feature.SetGeometry(geometry)                                      # Make a feature from geometry object
                layer.CreateFeature(feature)
                geometry = feature = None
                del x0, x1, y1, y0, id1
        return layer

# --- End Classes --- #

# --- Functions --- #

def get_projection_from_raster(in_raster):
    ''' Get projection from input raster and return.'''
    proj = osr.SpatialReference()
    proj.ImportFromWkt(in_raster.GetProjectionRef())
    return proj

def split_vertical(polygon, peices=2):
    '''Creates a specified number of clipping geometries which are boxes used to
    clip an OGR feature. Returns a list of geometry objects which are verticaly
    split chunks of the original polygon.'''

    tic = time.time()

    # Get polygon geometry information
    polygeom = polygon.GetGeometryRef()
    polygeom.CloseRings()                                                       # Ensure all rings are closed

    # Get min/max
    xmin, xmax, ymin, ymax = polygeom.GetEnvelope()                             # Get individual bounds from bounding envelope
    horizontal_dist = xmax - xmin                                               # Distance across the horizontal plane

    # Create clipping geometries
    clippolys = []           # List of new polygons
    interval = horizontal_dist/peices                                           # Split the horizontal distance using numsplits
    for split in range(peices):

        # Create clip-box bounds
        x0 = xmin+(split*interval)                                              # X-min - changes with section
        x1 = xmin+((split+1)*interval)                                          # X-max - changes with section
        y0 = ymin                                                               # Y-min - always the same
        y1 = ymax                                                               # Y-max - always the same

        # Create geometry for clip box
        myRing = ogr.Geometry(type=ogr.wkbLinearRing)
        myRing.AddPoint(x0, y1)
        myRing.AddPoint(x1, y1)
        myRing.AddPoint(x1, y0)
        myRing.AddPoint(x0, y0)
        myRing.AddPoint(x0, y1)                                                 #close ring
        geometry = ogr.Geometry(type=ogr.wkbPolygon)
        geometry.AddGeometry(myRing)

        # Add to the list of clipping geometries to be returned
        clippolys.append(geometry)
    return clippolys

def perform_intersection(gridder_obj, proj1, layer, fieldname):
    '''This function performs the intersection between two geometries.'''

    # Initiate counters
    counter = 0
    counter2 = 0

    # Test intersection with layer
    tic2 = time.time()
    spatialweights = {}                                                         # This yields the fraction of the key polygon that each overlapping polygon contributes
    regridweights = {}                                                          # This yields the fraction of each overlapping polygon that intersects the key polygon - for regridding
    other_attributes = {}                                                       # This dicitonary stores the i,j indices of the grid cells
    allweights = {}                                                             # This dictionary will store the final returned data

    # Set up coordinate transform from layer to the grid projection
    proj2 = layer.GetSpatialRef()
    coordTrans = osr.CoordinateTransformation(proj2, proj1)

    # Attempt using a grid layer returned by the gridder object
    print('  Layer feature count: {0: ^8}'.format(layer.GetFeatureCount()))
    for feature in layer:
        counter2 += 1
        id2 = feature.GetField(fieldname)
        geometry2 = feature.GetGeometryRef()
        geometry2.Transform(coordTrans)                                         # Transform the geometry from layer CRS to layer1 CRS
        polygon_area = geometry2.GetArea()

        # Check to find incompatible geometry types
        if check_geometry:
            if geometry2==None:
                print('  polygon {0} is NoneType'.format(id2))
                continue
            if not geometry2.IsValid():
                print('polygon {0} geometry invalid.'.format(id2))
                geometry2 = geometry2.Buffer(0)
                if not geometry2:
                    print('  polygon {0} is NoneType after Buffer operation.'.format(id2))
                    geometry2 = geometry2.Union(geometry2)
                    continue
                if not geometry2.IsValid():
                    print('  polygon {0} geometry not fixed by performing self-union or Buffer.'.format(id2))
                    continue

        # Split into parts
        if polygon_area > threshold:
            print('    Polygon: {0: ^8} area = {1: ^12}. Splitting into {2: ^2} sections.'.format(id2, polygon_area, splits))
            Areas = []
            inters = 0
            clip_polys = split_vertical(feature, splits)

            # Create temporary output polygon vector file to store the input feature
            drv1 = ogr.GetDriverByName('Memory')
            in_ds = drv1.CreateDataSource('in_ds')
            inlayer = in_ds.CreateLayer('in_ds', srs=proj1, geom_type=ogr.wkbPolygon)
            LayerDef = inlayer.GetLayerDefn()                                    # Fetch the schema information for this layer
            infeature = ogr.Feature(LayerDef)                                     # Create a new feature (attribute and geometry)
            infeature.SetGeometry(geometry2)                                       # Make a feature from geometry object
            inlayer.CreateFeature(infeature)

            for num,clipgeom in enumerate(clip_polys):
                tic3 = time.time()

                # Create temporary output polygon vector file
                out_ds = drv1.CreateDataSource('out_ds')
                outlayer = out_ds.CreateLayer('out_ds', srs=proj1, geom_type=ogr.wkbPolygon)
                outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))   # Create a new field on a layer. Add one attribute

                # Create temporary in-memory feature layer to store clipping geometry
                clip_ds = drv1.CreateDataSource('clip_ds')
                cliplayer = clip_ds.CreateLayer('clip_ds', srs=proj1, geom_type=ogr.wkbPolygon)
                LayerDef2 = cliplayer.GetLayerDefn()                                    # Fetch the schema information for this layer
                clipfeature = ogr.Feature(LayerDef2)                                     # Create a new feature (attribute and geometry)
                clipfeature.SetGeometry(clipgeom)                                       # Make a feature from geometry object
                cliplayer.CreateFeature(clipfeature)

                # Perform clip
                inlayer.Clip(cliplayer, outlayer)

                # Read clipped polygon feature
                assert outlayer.GetFeatureCount() == 1                      # Make sure the output has only 1 feature
                feat = outlayer.GetNextFeature()                            # The output should only have one feature
                geometry3 = feat.GetGeometryRef()

                # Create a Layer so that the SetSpatialFilter method can be used (faster for very large geometry2 polygons)
                drv = ogr.GetDriverByName('Memory')
                dst_ds = drv.CreateDataSource('out')
                gridlayer = dst_ds.CreateLayer('out', srs=proj1, geom_type=ogr.wkbPolygon)
                gridlayer = gridder_obj.getgrid(geometry3.GetEnvelope(), gridlayer)     # Generate the grid layer
                gridlayer.SetSpatialFilter(geometry3)                                   # Use the SetSpatialFilter method to thin the layer's geometry

                # Check to find incompatible geometry types
                #if check_geometry:
                #    keep_Going = True
                #    copyLayer = False
                #    doSQL = False
                #    for item in gridlayer:
                #        itemgeom = item.geometry()
                #        if itemgeom.Intersection(geometry3) is None:
                #            print('grid polygon %s intersection geometry invalid against polygon %s' %(item.GetField(0), id2))
                #            gridlayer.DeleteFeature(item.GetFID())
                #            #keep_Going = False
                #            copyLayer = True
                #            #doSQL = True
                #            continue
                #    if doSQL:
                #        #gridlayer = dst_ds.ExecuteSQL("select * from out WHERE ST_IsValid(geometry)", dialect = "SQLITE")  # Only select valid geometry
                #        gridlayer = dst_ds.ExecuteSQL("select ST_Buffer(geometry, 0), * from out" , dialect = "SQLITE") # Buffer with 0 distance to create valid geometry
                #    if copyLayer:
                #        driver = ogr.GetDriverByName('ESRI Shapefile')
                #        gridds = driver.CreateDataSource(os.path.join(OutDir, '%s.shp' %id2))
                #        griddslayer = gridds.CopyLayer(gridlayer, 'gridlayer')
                #        gridds = gridslayer = driver = None
                #    if not keep_Going:
                #        continue
                #    gridlayer.ResetReading()

                # First find all intersection areas
                Areas += [[item.GetField(0), item.geometry().Intersection(geometry3).Area(), item.geometry().Area(), item.GetField(1), item.GetField(2)] for item in gridlayer]  # Only iterate over union once
                inters += len(Areas)
                counter += inters                                                       # Advance the counter

                #flush memory
                clipfeature = clipgeom = geometry3 = feat = outlayer = cliplayer = clipfeature = None  # destroy these
                print('          Chunk {0: ^2}. Time elapsed: {1: ^4.2f} seconds.'.format(num+1, (time.time()-tic3)))

            # Collapse all duplicates back down to 1 list
            AreaDict = {}
            for item in Areas:
                try:
                    AreaDict[item[0]][1] += item[1]
                except KeyError:
                    AreaDict[item[0]] = item
            Areas = AreaDict.values()

        else:
            '''This is the normal case where polygons are smaller or more uniform in size.'''

            # Create a Layer so that the SetSpatialFilter method can be used (faster for very large geometry2 polygons)
            drv = ogr.GetDriverByName('Memory')
            dst_ds = drv.CreateDataSource('out')
            gridlayer = dst_ds.CreateLayer('out', srs=proj1, geom_type=ogr.wkbPolygon)
            gridlayer = gridder_obj.getgrid(geometry2.GetEnvelope(), gridlayer)     # Generate the grid layer
            gridlayer.SetSpatialFilter(geometry2)                                   # Use the SetSpatialFilter method to thin the layer's geometry

            ## Check to find incompatible geometry types
            #if check_geometry:
            #    keep_Going = True
            #    copyLayer = False
            #    doSQL = False
            #    for item in gridlayer:
            #        itemgeom = item.geometry()
            #        if itemgeom.Intersection(geometry2) is None:
            #            print('grid polygon %s intersection geometry invalid against polygon %s' %(item.GetField(0), id2))
            #            gridlayer.DeleteFeature(item.GetFID())
            #            #keep_Going = False
            #            copyLayer = True
            #            #doSQL = True
            #            continue
            #    if doSQL:
            #        #gridlayer = dst_ds.ExecuteSQL("select * from out WHERE ST_IsValid(geometry)", dialect = "SQLITE")  # Only select valid geometry
            #        gridlayer = dst_ds.ExecuteSQL("select ST_Buffer(geometry, 0), * from out", dialect = "SQLITE") # Buffer with 0 distance to create valid geometry
            #    if copyLayer:
            #        driver = ogr.GetDriverByName('ESRI Shapefile')
            #        gridds = driver.CreateDataSource(os.path.join(OutDir, '%s.shp' %id2))
            #        griddslayer = gridds.CopyLayer(gridlayer, 'gridlayer')
            #        gridds = griddslayer = driver = None
            #    if not keep_Going:
            #        continue
            #    gridlayer.ResetReading()

            # First find all intersection areas
            Areas = [[item.GetField(0), item.geometry().Intersection(geometry2).Area(), item.geometry().Area(), item.GetField(1), item.GetField(2)] for item in gridlayer]  # Only iterate over union once

            # Use the intersection area to thin the other lists
            inters = len(Areas)
            counter += inters                                                       # Advance the counter

        # Calculate area weights - for averaging
        spatialweights[id2] = [(item[0], (item[1]/polygon_area)) for item in Areas]

        # Calculate regrid weights - for conservative regridding
        regridweights[id2] = [(item[0], (item[1]/item[2])) for item in Areas]

        # Store i,j variables
        other_attributes[id2] = [[item[0], item[3], item[4]] for item in Areas]
        del gridlayer, Areas, inters, dst_ds, drv

    # Counter and printed information below
    print('      [{0: ^7} intersections processed in {1: ^4.2f} s] [{2: ^8.2f} features per second] [Processed {3: ^8} features in dest grid]'.format(counter, time.time()-tic2, (counter/(time.time()-tic2)), counter2))

    # print run information
    print('    Done gathering intersection information between layer 1 and layer 2 in {0: 8.2f} seconds'.format(time.time()-tic2))
    print('    {0: ^10} polygons processed for intersection with grid. {1: ^10} total polygon intersections processed.'.format(counter2, counter))
    allweights[0] = spatialweights
    allweights[1] = regridweights
    allweights[2] = other_attributes

    # Clean up and return
    return allweights

# --- End Functions --- #

# --- Main Codeblock --- #
if __name__ == '__main__':

    '''
    Below is a variety of example code that can be used in other scripts to harness
    the capabilities contained in the above funcitons.
    '''

    tic = time.time()
    print('Process initiated at %s' %time.ctime())

    # Code below will build the grid representation using an input raster
    if fromRaster:
        if os.path.exists(raster_path):
            print('Found path to raster file: {0}'.format(raster_path))

            # Opening the file with GDAL, with read only acces
            gdal.AllRegister()
            ds = gdal.Open(raster_path, gdal.GA_ReadOnly)

            ncols = ds.RasterXSize
            nrows = ds.RasterYSize
            print('        Input Raster Size: {0} x {1} x {2}'.format(ncols, nrows, ds.RasterCount))
            print('        Projection of input raster: {0}'.format(InRaster.GetProjection()))

            x00, DX, xskew, y00, yskew, DY = ds.GetGeoTransform()
            grid_proj4 = get_projection_from_raster(ds).ExportToProj4()
            ds = None
            del ds, xskew, yskew

        else:
            print('Could not find path to raster file: {0}'.format(raster_path))

    else:
        # create the spatial reference for the input grid
        grid_proj = osr.SpatialReference()
        grid_proj.ImportFromProj4(grid_proj4)

    # Some tests of functionality

    # Build the grid object and test the class functions
    gridObj = Gridder_Layer(DX, DY, x00, y00, nrows, ncols, grid_proj4)
    grid_proj = gridObj.proj

    gt = gridObj.GeoTransform()
    print(gt)

    grid_extent = gridObj.grid_extent()
    print(grid_extent)

    grid_envelope = gridObj.grid_envelope()
    print(grid_envelope)

    boundary_shape = gridObj.boundarySHP('', 'MEMORY')
    print(boundary_shape.Area())

    # Open the boundary polygon dataset to calculate spatial weights dynamically
    if use_poygons:

        driver = ogr.Open(polygons).GetDriver()
        ds_in = driver.Open(polygons, 0)
        inLayer = ds_in.GetLayer()                                              # Get the 'layer' object from the data source

        inLayer.SetAttributeFilter("{0} = '{1}'".format(fieldname, 'Colorado')) # Select the ID from the layer
        print('  After setting attribute filter: {0} features.'.format(inLayer.GetFeatureCount()))
        polygeom = ogr.Geometry(ogr.wkbMultiPolygon)

        # Initiate grider class object which will perform the gridding for each basin
        gridder_obj = Gridder_Layer(DX, -DY, x00, y00, nrows, ncols, grid_proj4)

        # Perform intersection between selected polygons and the grid
        allweights = perform_intersection(gridder_obj, grid_proj, inLayer, fieldname)

        # Clean up
        driver = inLayer = ds_in = polygeom = gridder_obj = None

    # Open spatial weight file to read pre-calculated spatial weights
    elif read_weights:
        if os.path.exists(in_nc):

            # Find if the spatial weights sum to 1
            rootgrp = netCDF4.Dataset(regridweightnc, 'r', format='NETCDF4_CLASSIC')
            variables = rootgrp.variables

            # Get the variables
            IDmask_arr = variables['IDmask'][:]
            i_index_arr = variables['i_index'][:]
            j_index_arr = variables['j_index'][:]

            # Read weights and IDs into arrays
            if spatialweight:
                weights_arr = variables['weight'][:]
            elif regridweight:
                weights_arr = variables['regridweight'][:]

    print('Process completed in %3.2f seconds' %(time.time()-tic))

# --- End Main Codeblock --- #