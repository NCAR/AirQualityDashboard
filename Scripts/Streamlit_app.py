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
# st.set_page_config(layout='wide')


# functions
# @st.cache
# def get_data(var):
#     return zarr.load(r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Zarr_Outputs\out_{}.zarr".format(var))


@st.cache
def get_geojson(geojson_path):
    return gpd.read_file(geojson_path)


# Main page text
st.title("Air Quality Index User Dashboard")

with st.expander("How to use this dashboard"):
    st.write("""
    1. Choose a time period from the sidebar\n
    2. Choose one or many tracers from the sidebar\n
    3. Choose the type of statistic used for aggregation \n
    4. Choose one or many geographic locations from the sidebar\n
    5. A download button will appear. Click to download the results in a CSV file.""")

with st.expander("What are the tracers?"):
    st.write("""
    AQI... \n
    Cloud fraction...\n
    CO...\n
    MDA8...\n
    NO2...\n
    pm10...\n
    pm1... \n
    pm25... \n
    SO2... \n
    SWDOWN ... \n
    T2 max ...\n
    T2 min ... \n
    """)


# Add map
st.header("Analyze geographic data")

# folium code
# with st.container():
#     states_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson")
#     counties_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson")
#     cities_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson")
#     geojson_option = st.selectbox('Choose data to display:', ('States', 'Counties', 'Cities'))
#     main_map = leafmap.Map(center=(38, -96), zoom=3, draw_control=False, measure_control=False)
#     main_map.add_basemap('SATELLITE')
#     main_map.add_basemap("HYBRID")
#     main_map.add_basemap('TERRAIN')
#     if geojson_option == 'States':
#         main_map.add_gdf(states_gdf, layer_name=geojson_option)
#     elif geojson_option == 'Cities':
#         main_map.add_gdf(cities_gdf, layer_name=geojson_option)
#     elif geojson_option == 'Counties':
#         main_map.add_gdf(counties_gdf, layer_name=geojson_option)
#     main_map.to_streamlit(add_layer_control=True)

# folium code v2
# @st.cache
# def open_geojsons():
#     states_geojson_in = r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson"
#     counties_geojson_in = r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson"
#     cities_geojson_in =r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson"
#     states = folium.GeoJson(data=(open(states_geojson_in, "r").read()), name='States')
#     counties = folium.GeoJson(data=(open(counties_geojson_in, "r").read()), name='Counties')
#     cities = folium.GeoJson(data=(open(cities_geojson_in, "r").read()), name='Cities')
#     return states, counties, cities
#
#
# states_geojson, counties_geojson, cities_geojson = open_geojsons


# with st.container():
#     states_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson")
#     counties_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson")
#     cities_gdf = get_geojson(
#         r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson")
#     geojson_option = st.selectbox('Choose data to display:', ('States', 'Counties', 'Cities'))
#     folium_map = folium.Map(tiles='Stamen Terrain', location=[38, -96], zoom_start=4, control_scale=True)
#     folium.TileLayer('OpenStreetMap').add_to(folium_map)
#
#     # folium.TileLayer(tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
#     #                  attr='Tiles courtesy of the <a href="https://usgs.gov/">U.S. Geological Survey</a>').add_to(folium_map)
#     # states_geojson = r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson"
#     # counties_geojson = r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson"
#     # cities_geojson =r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson"
#
#     geojson_dict = {"States": states_gdf, "Counties": counties_gdf, "Cities": cities_gdf}
#     geojson_layer = folium.GeoJson(
#         data=(geojson_dict[geojson_option]), name=geojson_option).add_to(folium_map)
#     folium.features.GeoJsonPopup(fields=['NAME'], labels=False).add_to(geojson_layer)
#     folium.LayerControl().add_to(folium_map)
#     folium_static(folium_map)


# Kepler code
with st.container():
    states_gdf = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_States.geojson")
    counties_gdf = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Counties.geojson")
    cities_gdf = get_geojson(
        r"C:\Users\casali\Documents\Projects\ForJenn\AirQualityDashboard\Data\Geojsons\US_Cities.geojson")
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

# Process data based on requests
