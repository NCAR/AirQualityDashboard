FROM python:3.9
WORKDIR /app

## Python installs
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/docker/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

#Set environment variables
ENV PATH="/home/docker/miniconda3/bin:${PATH}"

# Usage examples
RUN conda install -y -c conda-forge python pip numpy netcdf4 pandas dask xarray pyproj gdal geopandas matplotlib streamlit leafmap zarr keplergl && \
    conda clean -afy

# Try to get a clean streamlit install
RUN pip uninstall -y streamlit
RUN pip install streamlit

# Expose port for Streamlit
EXPOSE 8501

# Run `streamlit run streamlit_app.py`
#ENTRYPOINT [ "streamlit", "run" ]
#CMD ["Streamlit_app.py"]
# WORKS ! CMD ["streamlit", "run", "Streamlit_app.py"]
ENTRYPOINT ["streamlit", "run", "Streamlit_app.py"]
#CMD ["/bin/bash"]

