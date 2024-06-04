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
2022/06/27

    This script will store information relevant to any given grid. Designed to be
    imported from other scripts to define a particular model grid.
'''

# --- Import Modules --- #

# Import Python Core Modules
import os

# Import Additional Modules
import osgeo
from osgeo import gdal
from osgeo import osr

# --- End Import Modules --- #

# --- Global Variables --- #

# Obtain the path of this file, so other files may be found relative to it
file_path = os.path.abspath(os.path.dirname(__file__))

# A raster may be used to describe the model grid, rather than specifying with globals above
fromRaster = False
raster_path = os.path.abspath(os.path.join(file_path, r'..\Data\GIS\Tiff\pm25_daily.tif'))

# Air Quality grid information
gridName = 'AQ'
DX = 12000.0
DY = -12000.0
nrows = 265
ncols = 442
x00 = -2508000.256
y00 = 1464001.179
grid_proj4 = '+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs'

# --- End Global Variables --- #

# --- Functions --- #

def get_projection_from_raster(in_raster):
    ''' Get projection from input raster and return.'''
    proj = osr.SpatialReference()
    proj.ImportFromWkt(in_raster.GetProjectionRef())
    return proj

def grid_params():

    global DX, DY, nrows, ncols, x00, y00, grid_proj4

    # Code below will build the grid representation using an input raster
    if fromRaster:

        if os.path.exists(raster_path):

            print('Found path to template grid raster file: {0}'.format(raster_path))

            # Opening the file with GDAL, with read only acces
            gdal.AllRegister()
            ds = gdal.Open(raster_path, gdal.GA_ReadOnly)

            ncols = ds.RasterXSize
            nrows = ds.RasterYSize
            print('    Input Raster Size: {0} x {1} x {2}'.format(ncols, nrows, ds.RasterCount))
            print('    Projection of input raster: {0}'.format(ds.GetProjection()))

            x00, DX, xskew, y00, yskew, DY = ds.GetGeoTransform()
            grid_proj4 = get_projection_from_raster(ds).ExportToProj4()
            ds = None
            del ds, xskew, yskew

        else:
            print('Could not find path to raster file: {0}'.format(raster_path))

    return gridName, DX, DY, nrows, ncols, x00, y00, grid_proj4

# --- End Functions --- #

# --- Main Codeblock --- #
if __name__ == '__main__':

    gridName, DX, DY, nrows, ncols, x00, y00, grid_proj4 = grid_params()

# --- End Main Codeblock --- #
