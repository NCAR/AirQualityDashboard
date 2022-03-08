import streamlit as st
import pandas as pd
import zarr
import numpy as np


@st.cache
def get_data():
    return zarr.load('.Data/Zarr/testZarr.zip')


z = get_data()

st.header("Air Quality Index User Dashboard")
st.write("""
Description of how to use page \n
Description of what variables are
""")
test = str(z[0:1])
st.write(test)

