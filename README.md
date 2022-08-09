# AirQualityDashboard

To build a container image for this app, use:

`docker build -t airqualitydashboard:latest -f docker/Dockerfile .`

To run the app inside the container image, use:
  
  `docker run -p 8501:8501 -v <local path to directory containing repository>:/app -v <local path to directory containing input Zarr store>:/app/Data/Model_Data:ro -v <local path to directory containing scratch output directory>:/app/scratch airqualitydashboard:latest`