FROM condaforge/mambaforge:latest
WORKDIR /app

#Set environment variables 
#ENV PATH="/home/docker/miniconda3/bin:${PATH}"

# Copy environment file into container
COPY environment.yml /app

# Mamba commands
#RUN mamba env update --file environment.yml
RUN mamba install -y -c conda-forge python pip numpy netcdf4 pandas dask xarray pyproj gdal geopandas matplotlib streamlit leafmap zarr keplergl plotly
RUN mamba clean -afy

# Try to get a clean streamlit install
RUN pip uninstall -y streamlit
RUN pip install streamlit

# Expose port for Streamlit
EXPOSE 8501

# Run application
#CMD ["streamlit", "run", "Streamlit_app.py"] # Working
CMD ["streamlit", "run", "Streamlit_app.py", "--server.address=0.0.0.0"] 