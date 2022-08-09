FROM python:3.9
WORKDIR /app

## Python installs
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/docker/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

#Set environment variables
ENV PATH="/home/docker/miniconda3/bin:${PATH}"
 
# Install conda modules
RUN conda install -c conda-forge \
	numpy \
	netcdf4 \
	pandas \
	dask \
	xarray \
	pyproj \
	gdal \
	geopandas \
	matplotlib \
	streamlit \
	leafmap \
	zarr \
	keplergl

# Expose port for Streamlit
EXPOSE 8501

# Copy contents of entire repository to the app directory in the container
#COPY . /app

# Run `streamlit run streamlit_app.py`
ENTRYPOINT [ "streamlit", "run" ]
CMD ["Streamlit_app.py"]