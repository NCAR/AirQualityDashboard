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
import os
import sys
import time

# Import grid information from the grid_info.py script
from grid_tools import (main, plot_data, in_dataset, time_agg_dict)

# --- End Import Modules --- #

# --- Global Variables --- #

# BELOW IS FOR TESTING

# User selection items

# The zone dataset and feature name to test
vector_src_name = 'US_States.geojson'

# Zones should be passed as the ID and a label
#zone_names = {'08':'Colorado'} # '06':'California', '08':'Colorado'}
zone_names = {'06':'California'}

# Variable subset
#tracer_selections = ['AQI_daily', 'CO_daily_avg']
tracer_selections = ['CO', 'pm10']

# Time selection
starting_date = '2005-01-01'
ending_date = '2018-12-31'

# Time aggregation. Any pandas resample time period may be used.
# ['1A', '1M', '1W', '1D', 'W-SUN']
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
# Period start: ['AS', 'MS', 'WS', 'DS', 'W-SUN']
time_aggregation = 'Monthly'


# Statistical representation chosen: ['MEAN', 'MAX', 'MIN']
statistics = ['MEAN', 'MAX', 'MIN']

# ABOVE IS FOR TESTING

# --- End Global Variables --- #

# --- Main Codeblock --- #
if __name__ == '__main__':

    '''
    Below is a variety of example code that can be used in other scripts to harness
    the capabilities contained in the above funcitons.
    '''

    tic = time.time()
    print('Process initiated at %s' %time.ctime())

    # Process data
    zone_df_dict, out_files = main(
            in_dataset = in_dataset,
            start_time = starting_date,
            end_time = ending_date,
            resample_time_period = time_agg_dict[time_aggregation],
            Variables = tracer_selections,
            stats = statistics,
            vector_src_name=vector_src_name,
            zone_names = zone_names)

    # Create Plot
    save_plot = True
    show_plot = False
    if save_plot or show_plot:
        plt = plot_data(zone_names=zone_names,
                        zone_df_dict=zone_df_dict,
                        save_plot=save_plot)
        if show_plot:
            plt.show()
    print('Process completed in %3.2f seconds' %(time.time()-tic))

# --- End Main Codeblock --- #