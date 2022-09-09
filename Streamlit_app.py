"""
To do:
Add spatial data to zarr
Decide on zarr chunking
Combine all zarr data?

"""

import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import geopandas as gpd
import pyproj
import zarr
import numpy as np
import folium
import leafmap.kepler as leafmap
from datetime import datetime
import time
import zipfile

# Import grid information from the grid_info.py script
sys.path.insert(1, r'..\Scripts')
from Scripts import grid_info
from Scripts import spatial_weights
from Scripts import grid_tools
from grid_tools import (main, plot_data, in_dataset, time_agg_dict)

# Set wide mode
st.set_page_config(layout='wide')

# 9/8/2022 - Clear Cache
st.experimental_singleton.clear()

# --- Global Variables --- #

# Find the path to this directory
app_root = Path(__file__).parents[0]
out_dir = app_root / 'scratch'

# Name the data variables as they will appear in the UI and map to the variable name in the Zarr file
data_vars_dict = {'Air Quality Index (AQI)': 'AQI_daily',
                'Cloud fraction': 'Cloud_fraction_daily_avg',
                'CO mixing ratio': 'CO_daily_avg',
                'MDA8 ozone mixing ratio': 'MDA8_O3_daily',
                'PM10 mass concentrations': 'pm10_daily_avg',
                'PM1 mass concentrations': 'pm1_daily_avg',
                'PM2.5 mass concentrations': 'pm25_daily_avg',
                'SO2 mixing ratio': 'SO2_daily_avg',
                'Shortwave Down': 'SWDOWN_daily_total',
                'Air Temperature (2m) Daily Max': 'T2_max_daily',
                'Air Temperature (2m) Daily Min': 'T2_min_daily',
                'NO2 mixing ratios': 'NO2_daily_avg',}

data_vars_desc = '''
AQI_daily                : Daily Air Quality Index (AQI) based on O3, PM2.5, and PM10
CO_daily_avg             : Daily average CO mixing ratios
Cloud_fraction_daily_avg : Daily average Cloud fraction
MDA8_O3_daily            : Daily MDA8 ozone mixing ratios
NO2_daily_avg            : Daily average NO2 mixing ratios
SO2_daily_avg            : Daily average SO2 mixing ratios
SWDOWN_daily_total       : Daily total downward reaching solar radiation at the surface
T2_max_daily             : Daily maximum 2 m Temperature
T2_min_daily             : Daily minimum 2 m Temperature
pm10_daily_avg           : Daily average PM10 mass concentrations
pm1_daily_avg            : Daily average PM1 mass concentrations
pm25_daily_avg           : Daily average PM2.5 mass concentrations
'''

# functions

@st.cache
def get_geojson(geojson_path):
    return gpd.read_file(geojson_path)


# Alternatively, you could pull geometry from GitHub, but this is slower:
#    states = get_geojson("https://raw.githubusercontent.com/mcasali/AirQualityDashboard/master/Data/GIS/Boundaries/Geojsons/US_States.geojson")
@st.cache
def open_geojsons():
    states = get_geojson(str(spatial_weights.poly_dir / 'US_States.geojson'))
    counties = get_geojson(str(spatial_weights.poly_dir / 'US_Counties.geojson'))
    cities = get_geojson(str(spatial_weights.poly_dir / 'US_Cities.geojson'))
    return states, counties, cities

states_gdf, counties_gdf, cities_gdf = open_geojsons()

# Function to analyze the time series data
@st.cache
def analyze_data(in_dataset, starting_date, ending_date, time_aggregation, Variables, statistics, vector_src_name, zone_choice):
    # Process data
    zone_df_dict, out_files = main(
            in_dataset = in_dataset,
            start_time = starting_date,
            end_time = ending_date,
            resample_time_period = time_aggregation,
            Variables = Variables,
            stats = statistics,
            vector_src_name = vector_src_name,
            zone_names = zone_choice)
    return zone_df_dict, out_files

# Main page text
st.title("Air Quality Index User Dashboard")

with st.expander("How to use this dashboard"):
    st.write("""
    1. Choose a time period from the sidebar. The air quality data are available from 2005-2019. \n
    2. Select a time aggregation method. \n
    3. Choose one or many air quality tracers. \n
    4. Choose the type of statistic used for the data aggregation. \n
    5. Choose one or many geographic locations. Options include choosing an entire state, multiple counties, or urban areas.  \n
    6. A download button will appear. Click to download the results in CSV format.
    7. Additional plots will appear below the download area. These plots show previews of the selected data. """)

with st.expander("What are the tracers?"):
    st.write("""
    AQI: Maximum measure of air quality index from O3, PM, and other criteria pollutants \n
    Cloud fraction: Percentage of each pixel that is covered by clouds\n
    CO: Carbon monoxide concentration \n
    MDA8: Maxiumum daily 8-hour ozone average \n
    NO2: Nitrogen dioxide concentration \n
    PM10:  Particulate matter smaller than 10 µm in aerodynamic diameter \n
    PM1: Particulate matter smaller than 1 µm in aerodynamic diameter \n
    PM25: Particulate matter smaller than 25 µm in aerodynamic diameter \n
    SO2: Sulfur dioxide concentration \n
    SWDOWN: Downward shortwave radiation \n
    T2 Max: Maximum surface temperature \n
    T2 Min: Minimum sufarce temperature \n
    """)

with st.expander("Explore the Air Quality Data Dashboard Viewer"):
    st.write("Link to [Air Quality Dashboard Viewer](https://ncar.maps.arcgis.com/apps/dashboards/b7b795b14ed4428e953e558e180b6f75)")


# Add map
st.header("Analyze geographic data")


# Kepler code
with st.container():
    geojson_option = st.selectbox('Choose data to display:', ('States', 'Counties', 'Cities'))
    main_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
    geojson_dict = {"States": states_gdf, "Counties": counties_gdf, "Cities": cities_gdf}
    main_map.add_gdf(geojson_dict[geojson_option], layer_name=geojson_option, zoom_to_layer=False)
    main_map.to_streamlit()


# Add summary
st.header("Data request summary: \n")

# Sidebar
with st.container():
    st.sidebar.title("Selecting data:")

    # Choosing dates in sidebar
    st.sidebar.header("Select a time period: ")
    starting_date = st.sidebar.date_input("Starting date:", min_value=datetime(2005, 1, 1),
                                          max_value=datetime(2018, 12, 31), value=datetime(2005, 1, 1))
    st.write(f'Starting date is: {starting_date}')

    ending_date = st.sidebar.date_input("Ending date:", min_value=datetime(2005, 1, 1),
                                        max_value=datetime(2018, 12, 31), value=datetime(2018, 12, 31))
    st.write(f'Ending date is: {ending_date}')


    # Choosing time aggregation
    st.sidebar.header("Select a time aggregation: ")
    time_aggregation = st.sidebar.selectbox("Choose a time aggregation: ", ("Daily", "Weekly", "Monthly", "Yearly"))

    st.write(f"The selected time aggregation is: {time_aggregation}")


    # Choosing tracers
    st.sidebar.header("Select tracers: ")
    tracer_selections = st.sidebar.multiselect('Choose one or many tracers:', list(data_vars_dict.keys()))
    st.write(f"The tracers you have selected are: {', '.join(tracer_selections)}")


    # Choosing statistics
    st.sidebar.header("Select statistics: ")
    statistics = st.sidebar.multiselect('What statistic would you like calculated?', ('MEAN', 'MAX', 'MIN'))
    st.write(f"The statistics that will be calculated are: {', '.join(statistics)}")


    # Choosing geographic options
    st.sidebar.header("Select a geographic extent: ")
    geo_type = st.sidebar.radio("States/Counties or Cities:", ('States', 'States/Counties', 'Cities'))

    if geo_type == "States":
        state_list = states_gdf.NAME.to_list()
        state_choice = st.sidebar.multiselect('Choose a state:', sorted(state_list))
        vector_src_name = 'US_States.geojson'
        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
        zone_choice = {row[fieldname]:row['NAME'] for n,row in states_gdf.loc[states_gdf['NAME'].isin(state_choice)].iterrows()}
    elif geo_type == "States/Counties":
        state_list = states_gdf.NAME.to_list()
        state_choice = st.sidebar.selectbox('Choose a state:', sorted(state_list))
        counties_choice = st.sidebar.multiselect('Choose county/counties:',
                                                 sorted(counties_gdf.NAME.loc[counties_gdf.STATE == state_choice]))
        vector_src_name = 'US_Counties.geojson'
        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
        zone_choice = {row[fieldname]:row['NAME'] for n,row in counties_gdf.loc[counties_gdf['NAME'].isin(counties_choice)].iterrows()}
    elif geo_type == "Cities":
        cities_choice = st.sidebar.multiselect('Choose city/cities:', sorted(cities_gdf.NAME))
        vector_src_name = 'US_Cities.geojson'
        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
        zone_choice = {row[fieldname]:row['NAME'] for n,row in cities_gdf.loc[cities_gdf['NAME'].isin(cities_choice)].iterrows()}
    st.write("\t{0}: {1}".format(geo_type, list(zone_choice.values())))

##with st.form(key="Selecting data"):
##    st.sidebar.title("Selecting data:")
##
##    # Choosing dates in sidebar
##    st.sidebar.header("Select a time period: ")
##    starting_date = st.sidebar.date_input("Starting date:", min_value=datetime(2005, 1, 1),
##                                          max_value=datetime(2018, 12, 31), value=datetime(2005, 1, 1))
##    st.write(f'Starting date is: {starting_date}')
##
##    ending_date = st.sidebar.date_input("Ending date:", min_value=datetime(2005, 1, 1),
##                                        max_value=datetime(2018, 12, 31), value=datetime(2018, 12, 31))
##    st.write(f'Ending date is: {ending_date}')
##
##
##    # Choosing time aggregation
##    st.sidebar.header("Select a time aggregation: ")
##    time_aggregation = st.sidebar.selectbox("Choose a time aggregation: ", ("Daily", "Weekly", "Monthly", "Yearly"))
##
##    st.write(f"The selected time aggregation is: {time_aggregation}")
##
##
##    # Choosing tracers
##    st.sidebar.header("Select tracers: ")
##    tracer_selections = st.sidebar.multiselect('Choose one or many tracers:', list(data_vars_dict.keys()))
##    st.write(f"The tracers you have selected are: {', '.join(tracer_selections)}")
##
##
##    # Choosing statistics
##    st.sidebar.header("Select statistics: ")
##    statistics = st.sidebar.multiselect('What statistic would you like calculated?', ('MEAN', 'MAX', 'MIN'))
##    st.write(f"The statistics that will be calculated are: {', '.join(statistics)}")
##
##
##    # Choosing geographic options
##    st.sidebar.header("Select a geographic extent: ")
##    geo_type = st.sidebar.radio("States/Counties or Cities:", ('States', 'States/Counties', 'Cities'))
##
##    if geo_type == "States":
##        state_list = states_gdf.NAME.to_list()
##        state_choice = st.sidebar.multiselect('Choose a state:', sorted(state_list))
##        vector_src_name = 'US_States.geojson'
##        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
##        zone_choice = {row[fieldname]:row['NAME'] for n,row in states_gdf.loc[states_gdf['NAME'].isin(state_choice)].iterrows()}
##    elif geo_type == "States/Counties":
##        state_list = states_gdf.NAME.to_list()
##        state_choice = st.sidebar.selectbox('Choose a state:', sorted(state_list))
##        counties_choice = st.sidebar.multiselect('Choose county/counties:',
##                                                 sorted(counties_gdf.NAME.loc[counties_gdf.STATE == state_choice]))
##        vector_src_name = 'US_Counties.geojson'
##        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
##        zone_choice = {row[fieldname]:row['NAME'] for n,row in counties_gdf.loc[counties_gdf['NAME'].isin(counties_choice)].iterrows()}
##    elif geo_type == "Cities":
##        cities_choice = st.sidebar.multiselect('Choose city/cities:', sorted(cities_gdf.NAME))
##        vector_src_name = 'US_Cities.geojson'
##        fieldname = spatial_weights.vector_fieldmap[vector_src_name]
##        zone_choice = {row[fieldname]:row['NAME'] for n,row in cities_gdf.loc[cities_gdf['NAME'].isin(cities_choice)].iterrows()}
##    st.write("\t{0}: {1}".format(geo_type, list(zone_choice.values())))
##
##
##    plot_it = st.checkbox("Plot Data")
##    if starting_date and ending_date and tracer_selections and statistics:
##        submit = st.form_submit_button(label='Submit')
##    if submit:
##        with st.spinner('Exporting data...'):
##
##            # Process data
##            st.write('Process initiated at %s' %time.ctime())
##            tic = time.time()
##            zone_df_dict, out_files = analyze_data(in_dataset,
##                                                    starting_date,
##                                                    ending_date,
##                                                    time_agg_dict[time_aggregation],
##                                                    [data_vars_dict[tracer] for tracer in tracer_selections],
##                                                    statistics,
##                                                    vector_src_name,
##                                                    zone_choice)
##            if plot_it:
##                # Create Plot
##                tic = time.time()
##                import matplotlib.pyplot as plt
##                plt = plot_data(zone_names=zone_choice,
##                                zone_df_dict=zone_df_dict,
##                                save_plot=False,
##                                stats=statistics)
##                st.pyplot(plt)
##                st.write('Plot generated in %3.2f seconds' %(time.time()-tic))
##
##
##            # Zip up files if more than one CSV is output
##            if len(out_files) > 1:
##                zipped_file = out_dir / 'AQ_{0}.zip'.format(time.strftime('%Y-%m-%d_%H%M%S'))
##                with zipfile.ZipFile(zipped_file, 'w') as f:
##                    for file in out_files:
##                        f.write(file, os.path.basename(file))
##
##                with open(str(zipped_file), "rb") as fp:
##                    btn = st.download_button(
##                        label="Download data as CSVs in ZIP format",
##                        data=fp,
##                        file_name=zipped_file.name,
##                        mime="application/zip"
##                    )
##
##            else:
##                out_file = out_files[0]
##                with open(out_file, "r") as fp:
##                    btn = st.download_button(
##                        label="Download data as CSV",
##                        data=fp,
##                        file_name=os.path.basename(out_file),
##                        mime="text/plain",
##                        )
##
##        st.success('Done!')

# Add button to query data
with st.container():
    if st.sidebar.button("Run Query for data download"):
        if starting_date and ending_date and tracer_selections and statistics:
            with st.spinner('Exporting data...'):

                # Process data
                st.write('Process initiated at %s' %time.ctime())
                tic = time.time()
                zone_df_dict, out_files = analyze_data(in_dataset,
                                                        starting_date,
                                                        ending_date,
                                                        time_agg_dict[time_aggregation],
                                                        [data_vars_dict[tracer] for tracer in tracer_selections],
                                                        statistics,
                                                        vector_src_name,
                                                        zone_choice)
                st.write('Process completed in {0:3.2f} seconds'.format(time.time()-tic))

                # Zip up files if more than one CSV is output
                if len(out_files) > 1:
                    zipped_file = out_dir / 'AQ_{0}.zip'.format(time.strftime('%Y-%m-%d_%H%M%S'))
                    with zipfile.ZipFile(zipped_file, 'w') as f:
                        for file in out_files:
                            f.write(file, os.path.basename(file))

                    with open(str(zipped_file), "rb") as fp:
                        btn = st.download_button(
                            label="Download data as CSVs in ZIP format",
                            data=fp,
                            file_name=zipped_file.name,
                            mime="application/zip"
                        )

                else:
                    out_file = out_files[0]
                    with open(out_file, "r") as fp:
                        btn = st.download_button(
                            label="Download data as CSV",
                            data=fp,
                            file_name=os.path.basename(out_file),
                            mime="text/plain",
                            )
            st.success('Done!')

        else:
            st.sidebar.write("Please finish selecting data")

# Add a plot to the existing data
with st.container():
    if st.sidebar.button("Plot Selection"):
        if starting_date and ending_date and tracer_selections and statistics:
            with st.spinner('Processing data...'):

                # Process data
                st.write('Process initiated at %s' %time.ctime())
                tic = time.time()
                zone_df_dict, out_files = analyze_data(in_dataset,
                                                        starting_date,
                                                        ending_date,
                                                        time_agg_dict[time_aggregation],
                                                        [data_vars_dict[tracer] for tracer in tracer_selections],
                                                        statistics,
                                                        vector_src_name,
                                                        zone_choice)
                st.write('Process completed in {0:3.2f} seconds'.format(time.time()-tic))

                # Create Plot
                tic = time.time()
                import matplotlib.pyplot as plt
                plt = plot_data(zone_names=zone_choice,
                                zone_df_dict=zone_df_dict,
                                save_plot=False,
                                stats=statistics)
                st.pyplot(plt)
                st.write('Plot generated in %3.2f seconds' %(time.time()-tic))

        else:
            st.sidebar.write("Please finish selecting data")