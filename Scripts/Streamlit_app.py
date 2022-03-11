import streamlit as st
import pandas as pd
import geopandas as gpd
import zarr
import numpy as np


@st.cache
def get_data():
    return pd.read_parquet('https://github.com/mcasali/AirQualityDashboard/blob/main/Data/Testing/test.parquet',
                           engine="fastparquet")

@st.cache
def get_states():
    states = gpd.read_file('https://github.com/mcasali/AirQualityDashboard/blob/main/Data/GIS/Boundaries/States/cb_2018_us_state_5m.shp')
    return states

# df = get_data()

st.header("Air Quality Index User Dashboard")
st.write("""
Description of how to use page \n
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

state_data = get_states()
st.map(state_data)
