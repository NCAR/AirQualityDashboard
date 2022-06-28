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
import numpy as np
import xarray as xr

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

# Import grid information from the grid_info.py script
import grid_info
gridName, DX, DY, nrows, ncols, x00, y00, grid_proj4 = grid_info.grid_params()
from spatial_weights import (Gridder_Layer,
                                poly_ext,
                                checkfield,
                                intersection_compute,
                                getfieldinfo,
                                write_spatial_weights,
                                weight_grid)

# --- End Import Modules --- #

# --- Global Variables --- #

# Obtain the path of this file, so other files may be found relative to it
file_path = os.path.abspath(os.path.dirname(__file__))

# Output directory
#OutDir = file_path
OutDir = r'C:\Users\ksampson\Desktop\Air_Quality_Scratch'

# A raster may be used to describe the model grid, rather than specifying with globals above
fromRaster = False
raster_path = os.path.join(file_path, r'..\Data\GIS\Tiff\pm25_daily.tif')

# Specify polygon file for calculating boundaries
use_poygons = True
vector_src = os.path.join(file_path, r'..\Data\GIS\Boundaries\Geojsons\US_States.geojson')
#vector_src = os.path.join(file_path, r'..\Data\GIS\Boundaries\Geojsons\US_Counties.geojson')
#vector_src = os.path.join(file_path, r'..\Data\GIS\Boundaries\Geojsons\US_Cities.geojson')
fieldname = 'NAME'

# Use pre-calculated spatial weight file?
read_weights = True
regridweightnc = r'C:\Users\ksampson\Documents\GitHub\AirQualityDashboard\Data\GIS\spatial_weights\{grid}_{filename}_spatialweights.nc'

# Other options
weight_dtype = 'f8'                                                             # Numpy dtype for the spatial weights
RasterDriver = 'GTiff'                                                          # Raster output format

# Zarr file (input model results)
#in_dataset = os.path.join(file_path, r'..\Data\Zarr\array_XS.zarr')

# Attributes specific to underlying data store
xVar = 'x'
yVar = 'y'
latVar = 'latitude'
lonVar = 'longitude'
timeVar = 'time'

# BELOW IS FOR TESTING
#in_dataset = r'C:\Data\Projects\Air_Quality\data\AQ_Tracer_Data_SM_time_0_1.nc'
in_dataset = r'C:\Data\Projects\Air_Quality\data\AQ_Tracer_Data_SM.zarr'

zone_names = ['Colorado']

# Output spatial weight raster (for viewing and confirming orientation, etc.)
OutGTiff_template =  r'C:\Users\ksampson\Desktop\Air_Quality_Scratch\spatialweights_{zone_name}.tif'

# Use a grid-to-basin weighting function (spatialweight) or basin-to-grid (regridweight)
spatialweight = True
regridweight = False

# --- End Global Variables --- #

# --- Functions --- #

def read_Zarr():
    '''
    '''
    tic1 = time.time()


def subset_DS_spatial(ds, top, bottom, left, right):
    '''
    '''
    tic1 = time.time()

    top = 40
    bottom = 37
    left = 258
    right = 265.4


    ds_sel = ds.isel(lon=(ds.lon >= left) & (ds.lon <= right),
                              lat=(ds.lat >= bottom) & (ds.lat <= top),
                              )
    ds_sel_avg = ds_sel.mean(dim=['lat','lon'])

def main(in_file = in_dataset,
            start_date = None,
            end_date = None,
            time_agg = '1D',
            variables = [],
            statistics = [],
            geog_extent = []):
    '''
    Main function to handle inputs from user interface and call functionality
    from this script.

    Inputs:
        -
    Outputs:
        -
    '''

    # Open input dataset
    #ds = xr.open_dataset(in_file)
    ds = xr.open_zarr(in_file)

    # --- Subset by variables --- #

    # Drop all other variables except time and the requested variable
    keep_vars = [timeVar, 'crs', 'crs_t', 'x', 'y', 'latitude', 'longitude',]
    drop_vars = [var_in for var_in in ds.variables if var_in not in varNames+keep_vars]
    print('Dropping {0} from input file.'.format(drop_vars))
    ds = ds.drop(drop_vars)

    # Subset in space
    # This might be handled by the spatial weighting grid

    # Subset in time
    start_time = ds['Time'][0]
    end_time = ds['Time'][-1]
    ds = ds.sel(Time=slice(start_time, end_time))

    # Aggregate over time if requested
    time_agg = "1D"
    ds = ds[variables].resample({timeVar:time_agg}).mean(dim=timeVar) # , closed='right')

    return ds

# --- End Functions --- #

# --- Main Codeblock --- #
if __name__ == '__main__':

    '''
    Below is a variety of example code that can be used in other scripts to harness
    the capabilities contained in the above funcitons.
    '''

    tic = time.time()
    print('Process initiated at %s' %time.ctime())

    # SOME TESTS OF FUNCTIONALITY

    # Build the grid object and test the class functions
    gridder_obj = Gridder_Layer(DX, DY, x00, y00, nrows, ncols, grid_proj4)

    # Specify the expected name of the regridding weight file
    #regridweightnc = os.path.join(weight_dir, '{0}_{1}_spatialweights.nc'.format(gridName, os.path.basename(vector_src).replace(poly_ext, '')))
    regridweightnc = regridweightnc.format(grid = gridName, filename = os.path.basename(vector_src).replace(poly_ext, ''))

    # Open spatial weight file to read pre-calculated spatial weights
    if read_weights and os.path.exists(regridweightnc):
        print('Found sptaial weight file: {0}'.format(regridweightnc))

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

    # Open the boundary polygon dataset to calculate spatial weights dynamically
    elif use_poygons:
        print('Dynamically creating spatial weights.')

        driver = ogr.Open(vector_src).GetDriver()
        ds_in = driver.Open(vector_src, 0)
        inLayer = ds_in.GetLayer()                                              # Get the 'layer' object from the data source

        # Check for existence of provided fieldnames
        field_defn, fieldslist = checkfield(inLayer, fieldname, vector_src)

        polygon_selected = False            # Use a selected polygon
        if polygon_selected:
            #inLayer.SetAttributeFilter("{0} = '{1}'".format(fieldname, zone_name)) # Select the ID from the layer
            inLayer.SetAttributeFilter("{0} in ({1})".format(fieldname, ','.join(zone_names))) # Select the ID from the layer
            print('  After setting attribute filter: {0} features.'.format(inLayer.GetFeatureCount()))

        # Perform intersection between selected polygons and the grid
        allweights = intersection_compute(fieldname = fieldname,
                                            gridder_obj = gridder_obj,
                                            inLayer = inLayer)
        spatialweights = allweights[0]
        regridweights = allweights[1]
        other_attributes = allweights[2]

        # Read weights and IDs into arrays
        if spatialweight:
            weightslist = [[item[1] for item in weight] for weight in spatialweights.values()]
            weights_arr = numpy.array([item for sublist in weightslist for item in sublist], dtype=numpy.object)
            del weightslist
        elif regridweight:
            rweightlist = [[item[1] for item in rweight] for rweight in regridweights.values()]
            weights_arr = numpy.array([item for sublist in rweightlist for item in sublist], dtype=numpy.object)
            del rweightlist

        # Create the array of ID masks for the weight array
        masklist = [[x[0] for y in x[1]] for x in spatialweights.items()]       # Get all the keys for each list of weights
        IDmask_arr = numpy.array([item for sublist in masklist for item in sublist], dtype=numpy.object)
        del masklist

        # Create array of the i and j indices
        iindexlist= [[item[1] for item in attribute] for attribute in other_attributes.values()]
        i_index_arr = np.array([item for sublist in iindexlist for item in sublist], dtype=np.object)
        jindexlist = [[item[2] for item in attribute] for attribute in other_attributes.values()]
        j_index_arr = np.array([item for sublist in jindexlist for item in sublist], dtype=np.object)
        del jindexlist, iindexlist, other_attributes, regridweights

        # Optionally write output spatial weight file (store for later)
        print('  Building spatial weight file: {0}'.format(regridweightnc))
        fieldtype1 = getfieldinfo(field_defn, fieldname)
        write_spatial_weights(regridweightnc, [allweights], fieldtype1)
        del allweights

        # Clean up
        driver = inLayer = ds_in = None

    # Create a grid of spatial weights
    weightgrid = True
    if weightgrid:
        for zone_name in zone_names:
            OutGTiff = OutGTiff_template.format(zone_name=zone_name)
            weight_grid(gridObj=gridder_obj,
                        OutGTiff=OutGTiff,
                        weights_arr=weights_arr,
                        i_index_arr=i_index_arr,
                        j_index_arr=j_index_arr,
                        IDmask_arr=IDmask_arr,
                        zone_name=zone_name)

    # Open the full dataset, dropping variables as necessary
    #drop_vars = [variable for variable in xr.open_zarr(file_input) if variable not in Variables]
    #ds_input = xr.open_zarr(in_dataset) # , drop_variables=drop_vars)   # chunks='auto'

    print('Process completed in %3.2f seconds' %(time.time()-tic))

# --- End Main Codeblock --- #