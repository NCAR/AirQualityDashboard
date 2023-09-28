FROM python:3.9
WORKDIR /app

# Install miniconda and update
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/docker/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

#Set environment variables 
ENV PATH="/home/docker/miniconda3/bin:${PATH}"

# Install conda modules
RUN conda update --all
#RUN conda update -c conda-forge --all
COPY environment.yml /app
RUN conda env update --file environment.yml

# Conda commands
#RUN conda update --all
#RUN conda install -y -c conda-forge mamba
#COPY environment.yml /app
#RUN mamba env update --file environment.yml
##RUN mamba install -y -c conda-forge python pip numpy netcdf4 pandas dask xarray pyproj gdal geopandas matplotlib streamlit leafmap zarr keplergl && \
##    mamba clean -afy

# Try to get a clean streamlit install
RUN pip uninstall -y streamlit
RUN pip install streamlit

# Expose port for Streamlit
EXPOSE 8501

# Run application
#ENTRYPOINT [ "streamlit", "run" ]
#CMD [ "Streamlist_app.py" ]
CMD ["streamlit", "run", "Streamlit_app.py"]
