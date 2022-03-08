import streamlit as st
import pandas as pd
import zarr
import numpy as np


@st.cache
def get_data():
    return zarr.load('.data/Zarr/array_XS.zarr')


z = get_data()

st.header("Air Quality Index User Dashboard")
st.write("""
Description of how to use page \n
Description of what variables are
""")
st.write(z)

