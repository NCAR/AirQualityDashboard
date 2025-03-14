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

    NOTE:
        For spatial weight files, grid indices are west-to-east (i) and
        south-to-north (j), starting in the lower left corner.
'''

# --- Import Modules --- #

# Import Python Core Modules
import time
import os
import sys
from distutils.version import LooseVersion
from functools import reduce
import zipfile
from pathlib import Path

# Import Additional Modules
import netCDF4
import numpy as np
import xarray as xr
import pandas as pd

# Import grid information from the grid_info.py script
path_root = Path(__file__).parents[0]
app_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import spatial_weights
import grid_info
gridName, DX, DY, nrows, ncols, x00, y00, grid_proj4 = grid_info.grid_params()

# --- End Import Modules --- #

# --- Global Variables --- #

# --- DO NOT EDIT BELOW THIS LINE --- #

# Specify the output directory and create it if necessary
out_dir = app_root / 'scratch'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Set path to the output directory. Must already exist and contain the in_dataset
data_dir = app_root / 'Data' / 'Model_Data'

# Zarr file (input model results)
#in_dataset = r'C:\Data\Projects\Air_Quality\data\AQ_Tracer_Data_SM_time_0_1.nc'
in_dataset = data_dir / 'AQ_Tracer_Data_SM.zarr'

# Variables to keep from input Zarr store (CF-compliance, etc)
keepVars = ['crs', 'crs_t']

# Attributes specific to underlying data store
xVar = 'x'
yVar = 'y'
latVar = 'latitude'
lonVar = 'longitude'
timeVar = 'Time'

# Output or input spatial weight file template
regridweightnc = os.path.abspath(spatial_weights.weight_dir / '{grid}_{filename}_spatialweights.nc')

# Use pre-calculated spatial weight file if possible?
# NOTE: To generate weights for all features, you can set read_weights=False, use_polygons=False, build_weights=True, and polygon_selected=False
read_weights = True             # If spatial weightfile exists, use it
use_poygons = True              # If spatial weightfile does not exist, dynamically generate weights
build_weights = False           # If dynamically computing weights, should we save them? This will only save features found in zone_names

# Use a grid-to-basin weighting function (spatialweight) or basin-to-grid (regridweight)
spatialweight = True
regridweight = False
weight_dtype = 'f8'                                                             # Numpy dtype for the spatial weights
RasterDriver = 'GTiff'                                                          # Raster output format

# Create a grid of spatial weights to test that the orientation of the weightfile is correct?
save_weight_grid = False

# Output spatial weight raster (for viewing and confirming orientation, etc.)
OutGTiff_template = os.path.abspath(out_dir / 'spatialweights_{zone_name}.tif')

# Dictionary to translate time aggregation selection to a useable pandas time aggregation function
time_agg_dict = {"Daily":'1D',
                    "Weekly":'W-SUN',
                    "Monthly":'MS',
                    "Yearly":'AS',}

# Save output files to Zip, if more than one output CSV is created. Only works in __main__
zipped_output = True

# --- End Global Variables --- #

# --- Functions --- #

def main(in_dataset = in_dataset,
            start_time = '2005-01-01',
            end_time = '2018-12-31',
            resample_time_period = '1M',
            Variables = [],
            stats = ['MEAN'],
            vector_src_name='US_States.geojson',
            zone_names = {}):
    '''
    Main function to handle inputs from user interface and call functionality
    from this script.

    Inputs:
        -
    Outputs:
        -
    '''
    # global regridweightnc

    # Save spatially aggregated time-series to CSV?
    save_out_df = False

    # Allow time aggregation?
    time_agg = True

    # Build the grid object and test the class functions
    gridder_obj = spatial_weights.Gridder_Layer(DX, DY, x00, y00, nrows, ncols, grid_proj4)

    # Specify the expected name of the regridding weight file
    vector_src = spatial_weights.vector_src_dict.get(vector_src_name, None)
    regridweightnc = os.path.abspath(spatial_weights.weight_dir / '{grid}_{filename}_spatialweights.nc')
    regridweightnc = regridweightnc.format(grid = gridName, filename = os.path.basename(vector_src).replace(spatial_weights.poly_ext, ''))

    # Open spatial weight file to read pre-calculated spatial weights
    if read_weights and os.path.exists(regridweightnc):
        IDmask_arr, weights_arr, i_index_arr, j_index_arr = read_weight_file(regridweightnc)

    # Open the boundary polygon dataset to calculate spatial weights dynamically
    elif use_poygons:
        IDmask_arr, weights_arr, i_index_arr, j_index_arr = generate_weights(vector_src,
                                                                                zone_names,
                                                                                gridder_obj,
                                                                                polygon_selected=True,
                                                                                build_weights=build_weights)

    else:
        print('No way to create spatial aggregation. Existing...')
        raise SystemExit

    # Open the full dataset, dropping variables as necessary
    drop_vars = [variable for variable in xr.open_zarr(in_dataset) if variable not in Variables+keepVars]
    ds_input = xr.open_zarr(in_dataset, drop_variables=drop_vars)

    # Isolate the data variables from the coordinate names
    data_vars = [varName for varName in ds_input.variables if varName not in ds_input.coords]

    # Subset by time
    ds_input = ds_input.sel({timeVar:slice(start_time, end_time)})

    # Subset in space using individual zones and perform spatial aggregation
    zone_ds_dict = {}
    for zone_name in zone_names:

        if save_weight_grid:
            OutGTiff = OutGTiff_template.format(zone_name=zone_name)
            weightRaster = True
        else:
            OutGTiff = None
            weightRaster = False

        # Get a 2D array of spatial weights for this zone
        weight_grid = spatial_weights.weight_grid(gridObj=gridder_obj,
                    OutGTiff=OutGTiff,
                    weights_arr=weights_arr,
                    i_index_arr=i_index_arr,
                    j_index_arr=j_index_arr,
                    IDmask_arr=IDmask_arr,
                    zone_name=zone_name,
                    weightRaster=weightRaster)

        # Important to transpose the weight grid so that the x,y order matches the Xarray DataSet dimension order.
        weight_grid = weight_grid.T

        # Create a DataArray object to store the weight grid
        weight_da = xr.DataArray(data=weight_grid, dims=[yVar, xVar])

        # Subset in space based on presence of non-zero weights
        ds_sub = ds_input.where(weight_da>0, drop=True)
        weight_da = weight_da.where(weight_da>0, drop=True)

        # Output to netCDF to make sure spatial component is correct
        #ds_sub.isel({timeVar:slice(0,2,None)}).to_netcdf(os.path.join(out_dir, '{0}.nc'.format(zone_name)))

        # Once we collapse the spatial dimensions, the projection varibles are no longer needed
        ds_sub = ds_sub.drop(keepVars)

        # Apply spatial groupby operation by multiplying the weights by the data
        ds_sub = (ds_sub*weight_da).sum(dim=[yVar,xVar])            # Sum the weighted values
        del weight_grid, weight_da

        zone_ds_dict[zone_name] = ds_sub
        del ds_sub

    # Perform temporal aggregation and save to output file
    zone_df_dict = {}
    out_files = []
    for zone_name, ds_output in zone_ds_dict.items():
        zone_label = zone_names[zone_name]

        # Loading the DataSet into memory now saves a little time later
        tic1 = time.time()
        ds_output.load()
        print('Loaded dataset in {0:3.2f} seconds.'.format(time.time()-tic1))

        # xarray.to_dataframe is very hungry and may be where all computaiton happens if everything is dask delayed before this
        tic1 = time.time()
        out_df = ds_output.to_dataframe()
        if save_out_df:
            out_df.to_csv(os.path.join(out_dir, 'AQ_{0}.csv'.format(zone_label)))
        print('To dataframe in {0:3.2f} seconds.'.format(time.time()-tic1))

        # Aggregate over time (resample). Doing this in pandas is much faster than in xarray.
        tic1 = time.time()
        if time_agg:
            out_df_list = []
            for i,stat_function in enumerate(stats):
                print('\tCalculating statistic: {0}.'.format(stat_function))
                if stat_function == 'MEAN':
                    out_df2 = out_df.resample(resample_time_period).mean()
                if stat_function == 'MAX':
                    out_df2 = out_df.resample(resample_time_period).max()
                if stat_function == 'MIN':
                    out_df2 = out_df.resample(resample_time_period).min()

                # Rename the column to append the statistical function and time aggregations
                out_df2 = out_df2.rename(columns={varName:'{0}_{1}_{2}'.format(varName, resample_time_period, stat_function) for varName in out_df2.columns})
                out_df_list.append(out_df2)
                del out_df2
            out_df3 = reduce(lambda x, y: pd.merge(x, y, on=timeVar), out_df_list)
            del out_df_list

        out_file = os.path.join(out_dir, 'AQ_{0}_{1}.csv'.format(resample_time_period, zone_label))
        out_df3.to_csv(out_file, float_format='%g')
        out_files += [out_file]
        zone_df_dict[zone_name] = out_df3
        del ds_output, out_df
        print('Final dataframe created in {0:3.2f} seconds.'.format(time.time()-tic1))

    # Clean up
    ds_input.close()
    del ds_input, zone_ds_dict
    return zone_df_dict, out_files

def plot_data(zone_names=[], zone_df_dict={}, save_plot=False, stats=[]):
    '''
    Plot the data generated by this script
    '''
    tic1 = time.time()
    import matplotlib.pyplot as plt

    # Get the appropriate number of colors for this number of districts
    blackandwhite = False # If only one time-series is present, use black and white
    if len(zone_names) == 1 and blackandwhite:
        colors = ['black']
    else:
        #colors = plt.cm.rainbow(np.linspace(0, 1, len(zone_names)))
        colors = plt.cm.Set1(np.linspace(0, 1, len(zone_names)))

    # Get list of variables to plot
    columns = [list(out_df.columns) for out_df in zone_df_dict.values()]
    columns = [item for sublist in columns for item in sublist]       # Flatten list
    var_roots = list(set(['_'.join(item.split('_')[0:-2]) for item in columns]))

    # Setup plot
    plt.close("all")
    fig = plt.figure()      # figsize=(20,10))
    fig.set_size_inches(20, 10*len(var_roots))

    # Iterate first over each data variable
    for n,var_root in enumerate(var_roots):
        print('Plotting variable: {0}'.format(var_root))

        # Plot options for this variable set
        ax = fig.add_subplot(len(var_roots),1,n+1)

        # Iterate over requested regions
        i = 0
        for zone_name, out_df in zone_df_dict.items():
            zone_label = zone_names[zone_name]

            # Find the columns to plot
            plot_cols = [column for column in out_df.columns if var_root in column]

            # Option to shade area between min and max value
            shaded = True
            if 'MAX' in stats and 'MIN' in stats and shaded:
                max_col = [item for item in plot_cols if item.endswith('MAX')][0]
                min_col = [item for item in plot_cols if item.endswith('MIN')][0]
                ax.fill_between(out_df.index,
                                out_df[max_col],
                                out_df[min_col],
                                color=colors[i],
                                alpha=0.3,
                                linewidth=0.0)
                plot_cols = [item for item in plot_cols if item not in [max_col, min_col]]

            for plot_col in plot_cols:

                if 'MEAN' in plot_col:
                    ax.plot(out_df[plot_col],
                            color=colors[i],
                            label='{0} (mean)'.format(zone_label)) # label='_nolegend_',
                else:
                    if 'MIN' in plot_col:
                        stat_label = 'minimum'
                    if 'MAX' in plot_col:
                        stat_label = 'maximum'
                    ax.plot(out_df[plot_col],
                            color=colors[i],
                            linestyle='dashed',
                            label='{0} ({1})'.format(zone_label, stat_label))  # color='grey', label='_nolegend_'
            i+=1

        ax.set_ylabel(var_root)
        ax.set_xlabel('Time')
        plt.margins(x=0)
        ax.legend(fontsize=14)
    #plt.tight_layout()

    if save_plot:
        tic2 = time.time()
        out_plot = os.path.join(out_dir, 'plot.png')
        plt.savefig(out_plot)
        print('Saved plot in {0}'.format(out_plot))

    print('Created plot in {0:3.2f} seconds.'.format(time.time()-tic1))
    return plt

def plot_data_plotly(zone_names=[], zone_df_dict={}, save_plot=False, stats=[]):
    '''
    Plot the data generated by this script
    '''
    tic1 = time.time()
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # define colors as a list
    #colors = px.colors.qualitative.Plotly  # Hex
    colors = px.colors.qualitative.D3       # Hex
    #colors = px.colors.qualitative.Set1    # RGB
    #colors = px.colors.qualitative.Vivid   # RGB

    # convert plotly hex colors to rgba to enable transparency adjustments
    def hex_rgba(hex, transparency):
        col_hex = hex.lstrip('#')
        col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
        col_rgb.extend([transparency])
        areacol = tuple(col_rgb)
        return areacol

    # convert rgb colors to rgba to enable transparency adjustments
    def rgb_rgba(rgb, transparency):
        return 'rgba'+rgb[3:-1]+f',{transparency})'

    rgba = [hex_rgba(c, transparency=0.5) for c in colors]
    colCycle = ['rgba'+str(elem) for elem in rgba]
    #colCycle = [rgb_rgba(c, transparency=0.6) for c in colors]

    # Make sure the colors run in cycles if there are more lines than colors
    def next_col(cols):
        while True:
            for col in cols:
                yield col
    line_color=next_col(cols=colCycle)

    # Get list of variables to plot
    columns = [list(out_df.columns) for out_df in zone_df_dict.values()]
    columns = [item for sublist in columns for item in sublist]       # Flatten list
    var_roots = list(set(['_'.join(item.split('_')[0:-2]) for item in columns]))

    # plotly  figure
    #fig = go.Figure()
    fig = make_subplots(rows=len(var_roots),
                        cols=1,
                        subplot_titles=var_roots,
                        vertical_spacing=0.1,
                        shared_xaxes=True,)

    # Get a new color for each of the regions
    new_cols = [next(line_color) for i in zone_df_dict]

    # Iterate first over each data variable
    for n,var_root in enumerate(var_roots):
        print('Plotting variable: {0}'.format(var_root))

        # Iterate over requested regions
        i = 0
        for zone_name, out_df in zone_df_dict.items():
            zone_label = zone_names[zone_name]

            # Find the columns to plot
            plot_cols = [column for column in out_df.columns if var_root in column]

            # Option to shade area between min and max value
            shaded = True
            if 'MAX' in stats and 'MIN' in stats and shaded:
                max_col = [item for item in plot_cols if item.endswith('MAX')][0]
                min_col = [item for item in plot_cols if item.endswith('MIN')][0]
                fig.append_trace(go.Scatter(x=out_df.index,
                                            y=out_df[min_col],
                                            line=dict(color='rgba(255,255,255,0)'),
                                            showlegend=False,
                                            legendgroup = i+1,
                                            legendgrouptitle_text=zone_label,), row=n+1, col=1)
                fig.append_trace(go.Scatter(x=out_df.index,
                                            y=out_df[max_col],
                                            fill='tonexty',
                                            fillcolor=new_cols[i],
                                            line=dict(color='rgba(255,255,255,0)'),
                                            showlegend=True,
                                            name='{0} {1} range'.format(zone_label, var_root),
                                            legendgroup = i+1,
                                            legendgrouptitle_text=zone_label,), row=n+1, col=1)

                plot_cols = [item for item in plot_cols if item not in [max_col, min_col]]

            for plot_col in plot_cols:

                if 'MEAN' in plot_col:
                    stat_label = 'mean'
                    fig.append_trace(go.Scatter(
                        x=out_df.index,
                        y=out_df[plot_col],
                        line=dict(color=new_cols[i], width=2.5),
                        name='{0} {1} {2}'.format(zone_label, var_root, stat_label),
                        showlegend=True,
                        legendgroup = i+1,
                        legendgrouptitle_text=zone_label,), row=n+1, col=1)
                else:
                    if 'MIN' in plot_col:
                        stat_label = 'minimum'
                    if 'MAX' in plot_col:
                        stat_label = 'maximum'
                    fig.append_trace(go.Scatter(
                        x=out_df.index,
                        y=out_df[plot_col],
                        line=dict(color=new_cols[i], width=2.5),
                        name='{0} {1} {2}'.format(zone_label, var_root, stat_label),
                        showlegend=True,
                        legendgroup = i+1,
                        legendgrouptitle_text=zone_label,), row=n+1, col=1)
            i+=1

    # Update the layout in different ways
    fig.update_layout(height=500*len(new_cols),
                      legend_tracegroupgap=0.5*len(new_cols),)
    fig.update_layout(legend=dict(groupclick="toggleitem"))

    # Try to get xaxis labels on each subplot
    fig.update_layout({'{0}_showticklabels'.format(axis):True for axis in fig.layout if axis.startswith('xaxis')})


    if save_plot:
        tic2 = time.time()
        out_plot = os.path.join(out_dir, 'plot.html')
        plotly.offline.plot(fig, filename=out_plot)
        print('Saved plot in {0}'.format(out_plot))

    print('Created plot in {0:3.2f} seconds.'.format(time.time()-tic1))
    return fig

def read_weight_file(regridweightnc):
    '''
    Read existing spatial weight file for spatial weights
    '''

    tic1 = time.time()
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

    rootgrp.close()
    del rootgrp ,variables
    print('Read spatial weight file in {0: 3.2f} seconds.'.format(time.time()-tic1))
    return IDmask_arr, weights_arr, i_index_arr, j_index_arr

def generate_weights(vector_src, zone_names, gridder_obj, polygon_selected=False, build_weights=False):
    '''
    '''

    tic1 = time.time()
    print('Dynamically creating spatial weights.')

    driver = ogr.Open(vector_src).GetDriver()
    ds_in = driver.Open(vector_src, 0)
    inLayer = ds_in.GetLayer()                                              # Get the 'layer' object from the data source

    # Check for existence of provided fieldnames
    fieldname = spatial_weights.vector_fieldmap.get(os.path.basename(vector_src), None)
    field_defn, fieldslist = spatial_weights.checkfield(inLayer, fieldname, vector_src)

    # Calculate the user-selected polygon's spatial weights (True) or calculate all weights (False)
    if polygon_selected:
        SQL = " or ".join(["{0} = '{1}'".format(fieldname, zone_name) for zone_name in zone_names])
        inLayer.SetAttributeFilter(SQL) # Select the ID from the layer
        print('  After setting attribute filter: {0} features.'.format(inLayer.GetFeatureCount()))

    # Perform intersection between selected polygons and the grid
    allweights = spatial_weights.intersection_compute(fieldname = fieldname,
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
    i_index_arr = np.array([item for sublist in iindexlist for item in sublist], dtype=object)
    jindexlist = [[item[2] for item in attribute] for attribute in other_attributes.values()]
    j_index_arr = np.array([item for sublist in jindexlist for item in sublist], dtype=object)
    del jindexlist, iindexlist, other_attributes, regridweights

    # Optionally write output spatial weight file (store for later)
    if build_weights:
        print('  Building spatial weight file: {0}'.format(regridweightnc))
        fieldtype1 = spatial_weights.getfieldinfo(field_defn, fieldname)
        spatial_weights.write_spatial_weights(regridweightnc, [allweights], fieldtype1)
    del allweights

    # Clean up
    driver = inLayer = ds_in = None
    print('Generated spatial weights in {0: 3.2f} seconds.'.format(time.time()-tic1))
    return IDmask_arr, weights_arr, i_index_arr, j_index_arr

# --- End Functions --- #

# --- Main Codeblock --- #
if __name__ == '__main__':

    '''
    Below is a variety of example code that can be used in other scripts to harness
    the capabilities contained in the above funcitons.
    '''

    tic = time.time()
    print('Process initiated at %s' %time.ctime())

    # BELOW IS FOR TESTING

    in_dataset = r'C:\Data\Projects\Air_Quality\data\AQ_Tracer_Data_SM.zarr'

    # User selection items

    # The zone dataset and feature name to test
    vector_src_name = 'US_States.geojson'
    #vector_src_name = 'US_Counties.geojson'
    #vector_src_name = 'US_Cities.geojson'

    # Zones should be passed as the ID and a label
    zone_names = {'06':'California', '08':'Colorado'}
    #zone_names = {'06037':'Los Angeles', '08031':'Denver'}
    #zone_names = {'51445':'Los Angeles--Long Beach--Anaheim, CA', '23527':'Denver--Aurora, CO'}

    # Variable subset
    tracer_selections = ['AQI_daily', 'CO_daily_avg']

    # Time selection
    starting_date = '2006-01-01'
    ending_date = '2018-01-01'

    # Time aggregation. Any pandas resample time period may be used.
    # ['1A', '1M', '1W', '1D', 'W-SUN']
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    # Period start: ['AS', 'MS', 'WS', 'DS', 'W-SUN']
    #resample_time_period = 'MS'
    time_aggregation = 'Monthly'

    # Statistical representation chosen: ['MEAN', 'MAX', 'MIN']
    statistics = ['MEAN', 'MAX', 'MIN']

    # Output directory
    #out_dir = r'C:\Users\ksampson\Desktop\Air_Quality_Scratch'

    # ABOVE IS FOR TESTING

    # Process data
    zone_df_dict, out_files = main(
            in_dataset = in_dataset,
            start_time = starting_date,
            end_time = ending_date,
            resample_time_period = time_agg_dict[time_aggregation],
            Variables = tracer_selections,
            stats = statistics,
            vector_src_name=vector_src_name,
            zone_names=zone_names)


    # TEST TO SAVE MULTIPLE CSVs TO ZIP
    if zipped_output:
        zipped_file = os.path.join(out_dir, 'AQ_{0}.zip'.format(time.strftime('%Y-%m-%d_%H%M%S')))
        with zipfile.ZipFile(zipped_file, 'w') as f:
            for file in out_files:
                f.write(file, os.path.basename(file))


    # Create Plot
    save_plot = False
    show_plot = True
    if save_plot or show_plot:
        plt = plot_data(zone_names=zone_names,
                        zone_df_dict=zone_df_dict,
                        save_plot=save_plot,
                        stats=statistics)
        if show_plot:
            plt.show()
    print('Process completed in {0:3.2f} seconds'.format(time.time()-tic))

# --- End Main Codeblock --- #