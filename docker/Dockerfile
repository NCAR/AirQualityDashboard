FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt

## Python installs
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/docker/miniconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh

#Set environment variables
ENV PATH="/home/docker/miniconda3/bin:${PATH}"

#RUN apt-get update && apt-get install -y \
#  python3-pip \
#  binutils \
#  gdal-bin \
#  python3-gdal
  
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
	
#RUN pip3 install -r requirements.txt

EXPOSE 8501
COPY . /app
ENTRYPOINT [ "streamlit", "run" ]
CMD ["Streamlit_app.py"]