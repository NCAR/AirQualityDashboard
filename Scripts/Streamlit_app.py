"""
To do:
Add spatial data to zarr
Decide on zarr chunking
Combine all zarr data?

"""


import streamlit as st
import pandas as pd
import geopandas as gpd
import pyproj
import zarr
import numpy as np
import folium
import leafmap.foliumap as leafmap
import leafmap.kepler as leafmap
from datetime import datetime
import time
from streamlit_folium import folium_static


# Set wide mode
st.set_page_config(layout='wide')


# functions
# @st.cache
# def get_data(var):
#     return zarr.load(r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Zarr_Outputs\out_{}.zarr".format(var))


# @st.cache
@st.experimental_memo
def get_geojson(geojson_path):
    return gpd.read_file(geojson_path)


# @st.cache
@st.experimental_memo
def open_geojsons():
    states = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson")
    counties = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson")
    cities = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson")
    return states, counties, cities


states_gdf, counties_gdf, cities_gdf = open_geojsons()

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


@st.experimental_singleton
def create_state_map():
    new_state_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
    new_state_map.add_gdf(states_gdf, layer_name='States', zoom_to_layer=False)
    return new_state_map


@st.experimental_singleton
def create_county_map():
    new_county_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
    new_county_map.add_gdf(counties_gdf, layer_name='Counties', zoom_to_layer=False)
    return new_county_map


@st.experimental_singleton
def create_cities_map():
    new_cities_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
    new_cities_map.add_gdf(cities_gdf, layer_name='Cities', zoom_to_layer=False)
    return new_cities_map


@st.experimental_singleton
def create_maps():
    state_map_init = create_state_map()
    county_map_init = create_county_map()
    cities_map_init = create_cities_map()
    return state_map_init, county_map_init, cities_map_init


# Kepler code
with st.container():
    geojson_option = st.selectbox('Choose data to display:', ('States', 'Counties', 'Cities'))
    main_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
    geojson_dict = {"States": states_gdf, "Counties": counties_gdf, "Cities": cities_gdf}
    main_map.add_gdf(geojson_dict[geojson_option], layer_name=geojson_option, zoom_to_layer=False)
    main_map.to_streamlit()


    # state_map, county_map, cities_map = create_maps()
    # if geojson_option == 'States':
    #     state_map.to_streamlit()
    # elif geojson_option == 'Counties':
    #     county_map.to_streamlit()
    # elif geojson_option == 'Cities':
    #     cities_map.to_streamlit()



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
    tracer_selections = st.sidebar.multiselect('Choose one or many tracers:', ['AQI', 'Cloud fraction', 'CO', 'MDA8',
                                                                               'pm10', 'pm1', 'pm25', 'SO2', 'SWDOWN',
                                                                               'T2 max', 'T2 min'])
    st.write(f"The tracers you have selected are: {', '.join(tracer_selections)}")


    # Choosing statistics
    st.sidebar.header("Select statistics: ")
    statistics = st.sidebar.multiselect('What statistic would you like calculated?', ('MAX', 'MIN', 'MEAN'))
    st.write(f"The statistics that will be calculated are: {', '.join(statistics)}")


    # Choosing geographic options
    st.sidebar.header("Select a geographic extent: ")

    geo_type = st.sidebar.radio("States/Countiers or Cities:", ('States/Counties', 'Cities'))

    if geo_type == "States/Counties":
        state_list = states_gdf.NAME.to_list()
        state_choice = st.sidebar.selectbox('Choose a state:', sorted(state_list))
        counties_choice = st.sidebar.multiselect('Choose county/counties:',
                                                 sorted(counties_gdf.NAME.loc[counties_gdf.STATE == state_choice]))
    elif geo_type == "Cities":
        cities_choice = st.sidebar.multiselect('Choose city/cities:', sorted(cities_gdf.NAME))

    # st.write(f"The geographic areas are : {', '.join(county_choice)} in {state_choice}")


# Add button to query data
with st.container():
    if st.sidebar.button("Run query"):
        if starting_date and ending_date and tracer_selections and statistics:
            with st.spinner('Exporting data...'):
                time.sleep(2)
            st.success('Done!')

            st.download_button(
                label="Download data as CSV",
                data='',
                file_name='AirQuality.csv',
                mime='text/csv',
            )
        else:
            st.sidebar.write("Please finish selecting data")

