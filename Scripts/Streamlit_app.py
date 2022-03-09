import streamlit as st
import pandas as pd
import zarr
import numpy as np


@st.cache
def get_data():
    return pd.read_parquet('/Data/Parquet/test.parquet')


df = get_data()

st.header("Air Quality Index User Dashboard")
st.write("""
Description of how to use page \n
Description of what variables are
""")
test = str(df["station_id"][0])
st.write(test)

