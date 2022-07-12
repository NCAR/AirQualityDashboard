# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
# Copyright UCAR (c) 2020
# University Corporation for Atmospheric Research(UCAR)
# National Center for Atmospheric Research(NCAR)
# Research Applications Laboratory(RAL)
# P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
#
# Name:        module2
# Purpose:
# Author:      $ Kevin Sampson(ksampson)
# Created:     2022
# Licence:     <your licence>
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

'''
6/27/2022

    This script is designed to generate the spatial weight files from any arbitrary
    directory of spatial files (GeoJSON, Shapefile, etc.).

    This script should be called only if the spatial weight files have not yet been
    generated.
'''

# --- Import Modules --- #

# Import Python Core Modules
import sys
import os
import glob
import time
from pathlib import Path

# Import Additional Modules
import numpy as np
import netCDF4
import osgeo
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from osgeo.gdal_array import *                # Assists in using BandWriteArray, BandReadAsArray, and CopyDatasetInfo

# Import local functions and classes
path_root = Path(__file__).parents[0]
app_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import grid_info

# Import grid information from the grid_info.py script
gridName, DX, DY, nrows, ncols, x00, y00, grid_proj4 = grid_info.grid_params()

# --- End Import Modules --- #

# --- Global Variables --- #

# Specify the output directory and create it if necessary
out_dir = app_root / 'scratch'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Specify directory where polygon boundary files are stored
poly_dir = app_root / 'Data' / 'GIS' / 'Boundaries' / 'Geojsons'
poly_ext = r'.geojson'

# Specify directory to store spatial weight files
weight_dir = app_root / 'Data' / 'GIS' / 'spatial_weights'

# --- DO NOT EDIT BELOW THIS LINE --- #

# Run configurations
check_geometry = True                                                           # Check geometry to trap any .Area issues
threshold = 1e12                                                                # Number of square meters above which a polygon gets split up
splits = 2                                                                     # Number of chunks to divide the large polygon into
weight_dtype = 'f8'                                                             # Numpy dtype for the spatial weights
RasterDriver = 'GTiff'                                                          # Raster output format

# Specify polygon file for dynamically calculating spatial weights from polygon boundaries
vector_src_list = glob.glob(os.path.join(poly_dir, '*{0}'.format(poly_ext)))
vector_src_dict = {os.path.basename(key):key for key in vector_src_list}

# Map vector datasets to the appropriate fieldname for the unique ID of each feature
vector_fieldmap = {'US_Cities.geojson':'GEOID10',
                    'US_Counties.geojson':'GEOID',
                    'US_States.geojson':'GEOID'}

# Output vector driver name for boundary file ['ESRI Shapefile', 'MEMORY']
outDriverName = 'ESRI Shapefile'

# Write output boundary shapefile to disk?
grid_boundary = os.path.abspath(out_dir / '{0}_grid_boundary.shp'.format(gridName))

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
        #if int(osgeo.__version__[0]) >= 3:
        #    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        #    proj.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
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
                #feature.SetField('cellsize', geometry.Area())
                feature.SetField('i_index', x+1)
                feature.SetField('j_index', self.nrows-y)
                feature.SetGeometry(geometry)                                      # Make a feature from geometry object
                layer.CreateFeature(feature)
                geometry = feature = None
                del x0, x1, y1, y0, id1
        return layer

    def numpy_to_Raster(self, in_arr):
        '''
        This funciton takes in an input array. It will copy the input raster dataset
        information to the output such that they will have identical geotransform,
        coordinate system, and size. The output raster will have the data type of
        the input array. Statistics will be computed on the output.
        '''

        driver = gdal.GetDriverByName('Mem')
        gdaltype = NumericTypeCodeToGDALTypeCode(in_arr.dtype)
        print('    GDAL Data type derived from input array: {0} ({1})'.format(gdaltype, in_arr.dtype))

        outDS = driver.Create('', self.ncols, self.nrows, 1, gdaltype)
        outDS.SetProjection(self.proj.ExportToWkt())
        outDS.SetGeoTransform(self.GeoTransform())
        band = outDS.GetRasterBand(1)
        BandWriteArray(band, in_arr)
        stats = outDS.GetRasterBand(1).GetStatistics(0,1)                           # Calculate statistics
        driver = band = stats = None
        return outDS

# --- End Classes --- #

# --- Functions --- #

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

    # Allow for GDAL 3.0 changes to the order of coordinates in transform
    #if int(osgeo.__version__[0]) >= 3:
    #    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    #    proj2.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
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

                # First find all intersection areas
                Areas += [[item.GetField(0), item.geometry().Intersection(geometry3).Area(), item.geometry().Area(), item.GetField(1), item.GetField(2)] for item in gridlayer]  # Only iterate over union once
                inters += len(Areas)
                counter += inters                                                       # Advance the counter

                #flush memory
                clipfeature = clipgeom = geometry3 = feat = outlayer = cliplayer = clipfeature = None  # destroy these
                print('          Chunk {0: ^2}. Time elapsed: {1: ^3.2f} seconds.'.format(num+1, (time.time()-tic3)))

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

def checkfield(layer, fieldname, string1):
    '''Check for existence of provided fieldnames'''
    layerDefinition = layer.GetLayerDefn()
    fieldslist = []
    for i in range(layerDefinition.GetFieldCount()):
        fieldslist.append(layerDefinition.GetFieldDefn(i).GetName())
    if fieldname in fieldslist:
        i = fieldslist.index(fieldname)
        field_defn = layerDefinition.GetFieldDefn(i)
    else:
        print('    Field {0} not found in input {1}. Terminating...'.format(fieldname, string1))
        raise SystemExit
    return field_defn, fieldslist

def getfieldinfo(field_defn, fieldname):
    '''Get information about field type for buildng the output NetCDF file later'''
    if field_defn.GetType() == ogr.OFTInteger:
        fieldtype = 'integer'
        print("found ID type of Integer")
    elif field_defn.GetType() == ogr.OFTInteger64:
        fieldtype = 'integer64'
        print("found ID type of Integer64")
    elif field_defn.GetType() == ogr.OFTReal:
        fieldtype = 'real'
        print("field type: OFTReal not currently supported in output NetCDF file.")
        raise SystemExit
    elif field_defn.GetType() == ogr.OFTString:
        fieldtype = 'string'
        print("found ID type of String")
    else:
        print("ID Type not found ... Exiting")
        raise SystemExit
    print("    Field Type for field '%s': %s (%s)" %(fieldname, field_defn.GetType(), fieldtype))
    return fieldtype

def write_spatial_weights(regridweightnc, results, fieldtype1):
    '''Create a long-vector netCDF file. '''

    tic1 = time.time()
    NC_format = 'NETCDF4'

    # Get the size of the dimensions for constructing the netCDF file
    print('Beginning to get the size of the dictionaries.')
    dim1size = 0
    dim2size = 0
    counter = 1
    for allweights in results:
        dim1size += len(allweights[0])
        dim2size += sum([len(item) for item in allweights[0].values()])
        allweights = None
        counter += 1
    print('  Finished gathering dictionary length information')

    # variables for compatability with the code below, which was formerly from a function
    gridflag = 1
    print('Beginning to build weights netCDF file: {0} . Time elapsed: {1: 3.2f} seconds'.format(regridweightnc, time.time()-tic1))

    # Create netcdf file for this simulation
    rootgrp = netCDF4.Dataset(regridweightnc, 'w', format=NC_format)

    # Create dimensions and set other attribute information
    dim1name = 'polyid'
    dim2name = 'data'
    dim1 = rootgrp.createDimension(dim1name, dim1size)
    dim2 = rootgrp.createDimension(dim2name, dim2size)
    print('    Dimensions created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Handle the data type of the polygon identifier
    if fieldtype1 == 'integer':
        ids = rootgrp.createVariable(dim1name, 'i4', (dim1name))                # Coordinate Variable (32-bit signed integer)
        masks = rootgrp.createVariable('IDmask', 'i4', (dim2name))              # (32-bit signed integer)
    elif fieldtype1 == 'integer64':
        ids = rootgrp.createVariable(dim1name, 'i8', (dim1name))                # Coordinate Variable (64-bit signed integer)
        masks = rootgrp.createVariable('IDmask', 'i8', (dim2name))              # (64-bit signed integer)
    elif fieldtype1 == 'string':
        ids = rootgrp.createVariable(dim1name, str, (dim1name))                 # Coordinate Variable (string type character)
        masks = rootgrp.createVariable('IDmask', str, (dim2name))               # (string type character)
    print('    Coordinate variable created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Create fixed-length variables
    overlaps = rootgrp.createVariable('overlaps', 'i4', (dim1name))             # 32-bit signed integer
    weights = rootgrp.createVariable('weight', 'f8', (dim2name))                # (64-bit floating point)
    rweights = rootgrp.createVariable('regridweight', 'f8', (dim2name))         # (64-bit floating point)

    if gridflag == 1:
        iindex = rootgrp.createVariable('i_index', 'i4', (dim2name))            # (32-bit signed integer)
        jindex = rootgrp.createVariable('j_index', 'i4', (dim2name))            # (32-bit signed integer)
        iindex.long_name = 'Index in the x dimension of the raster grid (starting with 1,1 in LL corner)'
        jindex.long_name = 'Index in the y dimension of the raster grid (starting with 1,1 in LL corner)'
    print('    Variables created after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Set variable descriptions
    masks.long_name = 'Polygon ID (polyid) associated with each record'
    weights.long_name = 'fraction of polygon(polyid) intersected by polygon identified by poly2'
    rweights.long_name = 'fraction of intersecting polyid(overlapper) intersected by polygon(polyid)'
    ids.long_name = 'ID of polygon'
    overlaps.long_name = 'Number of intersecting polygons'
    print('    Variable attributes set after {0: 8.2f} seconds.'.format(time.time()-tic1))

    # Fill in global attributes
    rootgrp.history = 'Created %s' %time.ctime()

    # Iterate over dictionaries and begin filling in NC variable arrays
    dim1len = 0
    dim2len = 0
    counter = 1
    for allweights in results:
        tic2 = time.time()

        # Create dictionaries
        spatialweights = allweights[0].copy()
        regridweights = allweights[1].copy()
        other_attributes = allweights[2].copy()
        allweights = None

        # Set dimensions for this slice
        dim1start = dim1len
        dim2start = dim2len
        dim1len += len(spatialweights)
        dim2len += sum([len(item) for item in spatialweights.values()])

        # Start filling in elements
        if fieldtype1 == 'integer':
            ids[dim1start:dim1len] = np.array([x[0] for x in spatialweights.items()])    # Test to fix ordering of ID values
        if fieldtype1 == 'integer64':
            #ids[dim1start:dim1len] = np.array([x[0] for x in spatialweights.items()], dtype=np.long)    # Test to fix ordering of ID values
            ids[dim1start:dim1len] = np.array([x[0] for x in spatialweights.items()], dtype=np.int64)    # Test to fix ordering of ID values
        elif fieldtype1 == 'string':
            ids[dim1start:dim1len] = np.array([x[0] for x in spatialweights.items()], dtype=object)    # Test to fix ordering of ID values

        overlaps[dim1start:dim1len] = np.array([len(x) for x in spatialweights.values()])

        masklist = [[x[0] for y in x[1]] for x in spatialweights.items()]       # Get all the keys for each list of weights
        masks[dim2start:dim2len] = np.array([item for sublist in masklist for item in sublist], dtype=object)  # Flatten to 1 list (get rid of lists of lists)
        del masklist

        weightslist = [[item[1] for item in weight] for weight in spatialweights.values()]
        weights[dim2start:dim2len] = np.array([item for sublist in weightslist for item in sublist], dtype=object)
        del weightslist

        rweightlist = [[item[1] for item in rweight] for rweight in regridweights.values()]
        rweights[dim2start:dim2len] = np.array([item for sublist in rweightlist for item in sublist], dtype=object)
        del rweightlist

        if gridflag == 1:
            iindexlist= [[item[1] for item in attribute] for attribute in other_attributes.values()]
            iindex[dim2start:dim2len] = np.array([item for sublist in iindexlist for item in sublist], dtype=object)
            del iindexlist
            jindexlist = [[item[2] for item in attribute] for attribute in other_attributes.values()]
            jindex[dim2start:dim2len] = np.array([item for sublist in jindexlist for item in sublist], dtype=object)
            del jindexlist

        spatialweights = regridweights = other_attributes = None
        print('  [{0}] Done setting dictionary in {1: 3.2f} seconds.'.format(counter, time.time()-tic2))
        counter += 1

    # Close file
    rootgrp.close()
    del fieldtype1
    print('NetCDF correspondence file created in {0: 3.2f} seconds.'.format(time.time()-tic1))

def intersection_compute(fieldname = '',
                        gridder_obj = None,
                        inLayer = None):

    '''
    Helper function for outside scripts to compute spatial weights.
    '''
    # Perform intersection between selected polygons and the grid
    allweights = perform_intersection(gridder_obj, gridder_obj.proj, inLayer, fieldname)
    return allweights

def weight_grid(gridObj=None, OutGTiff=None, weights_arr=None, i_index_arr=None, j_index_arr=None, IDmask_arr=None, zone_name='', weightRaster=True):
    '''
    Given a weight array, create a grid of weight values. This is primarily used
    to test that weights are in the correct grid location and orientation for a
    given zone.
    '''
    print('  Raster will have nrows: {0} and ncols: {1}'.format(nrows, ncols))

    # Setup output driver
    driver = gdal.GetDriverByName('Memory')
    gdaltype = NumericTypeCodeToGDALTypeCode(numpy.dtype(weight_dtype))

    # Copy the empty raster
    weight_grid = np.zeros((ncols, nrows), dtype=weight_dtype)

    # Mask the array to just this basin ID
    basin_mask = IDmask_arr==zone_name

    #for weight,i,j in zip(weights_arr[basin_mask], i_index_arr-1, j_index_arr-1):
    for weight,i,j in zip(weights_arr[basin_mask], i_index_arr[basin_mask]-1, j_index_arr[basin_mask]-1):
        weight_grid[i,j] = weight

    # Output the grid of weights to a raster format (for confirming orientatin, etc.).
    if weightRaster:
        # Flip the y axis because NWM grid and indices are oriented south-north
        weight_grid = numpy.flip(weight_grid, axis=1)

        # Transpose so that order is y,x
        weight_grid = weight_grid.transpose()

        # Save to raster dataset
        OutRaster = gridObj.numpy_to_Raster(weight_grid)

        if OutRaster is not None:
            target_ds = gdal.GetDriverByName(RasterDriver).CreateCopy(OutGTiff, OutRaster)
            target_ds = None
        OutRaster = None
        print('Saved output weight raster to {0}'.format(OutGTiff))
    return weight_grid

def main(write_boundary = True):

    tic1 = time.time()
    print('Process initiated at %s' %time.ctime())

    # Build the grid object and test the class functions and caclulate spatial weights
    gridder_obj = Gridder_Layer(DX, DY, x00, y00, nrows, ncols, grid_proj4)

    gt = gridder_obj.GeoTransform()
    print('Grid GeoTransform: {0}'.format(gt))

    grid_extent = gridder_obj.grid_extent()
    print('Grid Extent: {0}'.format(grid_extent))

    grid_envelope = gridder_obj.grid_envelope()
    print('Grid Envelope: {0}'.format(grid_envelope))

    # Write output shapefile to disk
    if write_boundary:
        boundary_shape = gridder_obj.boundarySHP(grid_boundary, outDriverName)
        print('Total grid area: {0}'.format(boundary_shape.Area()))
        boundary_shape = None

    # Iterate over vector files on disk
    #vector_src_list = glob.glob(os.path.join(poly_dir, '*{0}'.format(poly_ext)))
    print('Found {0} vector sources in path {1}'.format(len(vector_src_list), poly_dir))

    for vector_src in vector_src_list:
        print('Creating spatial weights for vector source:\n\t{0}'.format(vector_src))

        # Generate output file name
        regridweightnc = os.path.join(weight_dir, '{0}_{1}_spatialweights.nc'.format(gridName, os.path.basename(vector_src).replace(poly_ext, '')))

        if os.path.exists(regridweightnc):
            print ('Output file already exists. Skipping...')
            continue

        # Open vector file
        driver = ogr.Open(vector_src).GetDriver()
        ds_in = driver.Open(vector_src, 0)
        inLayer = ds_in.GetLayer()                                              # Get the 'layer' object from the data source

        # Check for existence of provided fieldnames
        fieldname = vector_fieldmap.get(os.path.basename(vector_src), None)
        field_defn, fieldslist = checkfield(inLayer, fieldname, vector_src)

        # Perform intersection between selected polygons and the grid
        allweights = perform_intersection(gridder_obj, gridder_obj.proj, inLayer, fieldname)

        # Optionally write output spatial weight file (store for later)
        fieldtype = getfieldinfo(field_defn, fieldname)

        print('  Building spatial weight file: {0}'.format(regridweightnc))
        write_spatial_weights(regridweightnc, [allweights], fieldtype)
        del allweights

        # Clean up
        driver = inLayer = ds_in = None

    gridder_obj = None
    print('Computed spatial weights in %3.2f seconds' %(time.time()-tic1))

# --- End Functions --- #

# --- Main Codeblock --- #
if __name__ == '__main__':
    tic = time.time()
    main()
    print('Process completed in %3.2f seconds' %(time.time()-tic))
# --- End Main Codeblock --- #
