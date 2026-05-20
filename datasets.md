# EOL Datasets from M2HATS
Below are the datasets collected by EOL during M2HATS and currently hosted on EOL's [Field Data Archive](https://data.eol.ucar.edu/master_lists/generated/m2hats/). These datasets will be hosted on GDEX for this case study in the organization of EOL Field Campaign data using CISL resources and computing best practices. Please note that outside datasets supporting the campaign (METAR, ASOS, NCEP Stage IV data, etc.) are not included in this case study. 

## ISFS Surface meteorology and flux products
### isfs_m2hats_qc_geo_tiltcor_5min:
- **Sample rate:** 5 minutes
- **Coordinates:** Tilt corrected
- **Size:** 140 MB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** 
- **Status:** INCOMPLETE

### isfs_m2hats_qc_geo_hr_2023MM:
- **Sample rate:** 0.02 to 1 seconds
- **Coordinates:** Geographic
- **Size:** 252.723 GB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Read in all files together using open_mfdataset, then write to a Zarr store.
- **Status:** INCOMPLETE

### isfs_m2hats_qc_geo_tiltcor_hr_2023MM:
- **Sample rate:** 0.02 to 1 seconds
- **Coordinates:** Geographic with tilt corrected sonics
- **Size:** 252.559 GB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Read in all files together using open_mfdataset, then write to a Zarr store.
- **Status:** INCOMPLETE

## 915 MHz profiler datasets
- **Sample rate:** 30 minute averages
- **Processing:** (1) Standard and (2) with NIMA (NCAR Improved Moments Algorithm)
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## 449 MHz profiler datasets

## RASS datasets

## ISS Surface meteorology products

## Ceilometer Vaisala CL61 dataset

## Webcam imagery

## Lidar (Halo) datasets

## Lidar (Vaisala/Leosphere Windcube) datasets

## MicroPulse Differential Absorption Lidar (MPD) dataset

## ISS Radiosonde dataset
NetCDF-3 Classic