# EOL Datasets from M2HATS
Below are the datasets collected by EOL during M2HATS and currently hosted on EOL's [Field Data Archive](https://data.eol.ucar.edu/master_lists/generated/m2hats/). These datasets will be hosted on GDEX for this case study in the organization of EOL Field Campaign data using CISL resources and computing best practices. Please note that outside datasets supporting the campaign (METAR, ASOS, NCEP Stage IV data, etc.) are not included in this case study. 

## ISFS Surface meteorology and flux products
Each source product (named by its FDA file pattern below) is converted into a single Zarr v3 store (a DataTree with `array/` and `profile_t0/` groups, plus `soils/` and `radiation/` in the 5-minute product) in `/lustre/desc1/scratch/myasears/M2HATS/GDEX_datasets/`.

### isfs_m2hats_qc_geo_tiltcor_5min:
- **Sample rate:** 5 minutes
- **Coordinates:** Geographic with tilt corrected sonics
- **Size:** 140 MB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Write all netcdf files into a single Zarr store with centralized naming conventions and tower/height coordinates.  
- **GDEX store:** `isfs_m2hats_qc_geo_tiltcor_5min.zarr` (112 MB); built by [`notebooks/archive/ISFS_5min_tiltcor.ipynb`](notebooks/archive/ISFS_5min_tiltcor.ipynb) + [`code/m2hats_5min_to_zarr.py`](code/m2hats_5min_to_zarr.py)
- **Status:** Complete!

### isfs_m2hats_qc_geo_hr_2023MM:
- **Sample rate:** 0.02 to 1 seconds
- **Coordinates:** Geographic (NOT tilt corrected)
- **Size:** 252.723 GB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Write all netcdf files into a single Zarr store with centralized naming conventions and tower/height coordinates. 
- **GDEX store:** `isfs_m2hats_qc_geo_highrate.zarr` (197 GB); built by [`notebooks/archive/ISFS_highrate_geo.ipynb`](notebooks/archive/ISFS_highrate_geo.ipynb) + [`code/m2hats_to_zarr.py`](code/m2hats_to_zarr.py)
- **Status:** Complete!

### isfs_m2hats_qc_geo_tiltcor_hr_2023MM:
- **Sample rate:** 0.02 to 1 seconds
- **Coordinates:** Geographic with tilt corrected sonics
- **Size:** 252.559 GB
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Write all netcdf files into a single Zarr store with centralized naming conventions and tower/height coordinates. 
- **GDEX store:** `isfs_m2hats_qc_geo_tiltcor_highrate.zarr` (196 GB); built by [`notebooks/archive/ISFS_highrate_tiltcor.ipynb`](notebooks/archive/ISFS_highrate_tiltcor.ipynb) + [`code/m2hats_to_zarr.py`](code/m2hats_to_zarr.py)
- **Status:** Complete!

## 915 MHz profiler datasets
- **Sample rate:** 30 minute averages
- **Processing:** (1) Standard and (2) with NIMA (NCAR Improved Moments Algorithm)
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## 449 MHz profiler datasets
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## RASS datasets
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## ISS Surface meteorology products
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## Ceilometer Vaisala CL61 dataset
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## Webcam imagery
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## Lidar (Halo) datasets
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## Lidar (Vaisala/Leosphere Windcube) datasets
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## MicroPulse Differential Absorption Lidar (MPD) dataset
- **Sample rate:** 
- **Processing:** 
- **Size:** 
- **File type:** 
- **Proposed changes:** 
- **Status:** 

## ISS Radiosonde dataset
- **Sample rate:** 1 second; 122 Vaisala RS41 launches from ISS1 (2023-07-19 to 2023-09-25, mostly 16–17 and 21–22 UTC), with one file per ascent (`_asc`) and, for 115 launches, one per descent (`_dsc`)
- **Processing:** Aspen QC (`NCAR_M2HATS_ISS1_RS41_v1_*.nc`)
- **Size:** 59.9 MB (237 files)
- **File type:** NetCDF-3 Classic
- **Proposed changes:** Keep one NetCDF file per sounding rather than building a Zarr store: the dataset is small and already in the standard radiosonde format, so a Zarr would mainly save a few seconds of load time while dropping most per-sounding metadata (see the evaluation in [`notebooks/archive/ISS_radiosonde.ipynb`](notebooks/archive/ISS_radiosonde.ipynb)). The GDEX copy:
  - corrects the descent timestamps: in the FDA files, descent times are seconds since launch but referenced to balloon burst, so every descent is stamped late by the length of its ascent (57–102 min). The reference is now the ascent's launch time; stored values are unchanged. This error should also be reported to EOL.
  - renames variables to match the ISFS Zarr stores (e.g. `tdry` → `air_temperature`, `u_wind` → `wind_u`, `pres` → `air_pressure`), with ISFS-style units; the source name and units are kept in `source_short_name` / `source_units`.
  - is written as zlib-compressed NetCDF-4.
- **GDEX files:** `iss1_m2hats_rs41_radiosonde/` (237 files, 56.6 MB); built by [`notebooks/archive/ISS_radiosonde.ipynb`](notebooks/archive/ISS_radiosonde.ipynb) + [`code/m2hats_radiosonde.py`](code/m2hats_radiosonde.py). Example: [`examples/radiosonde_LCL_case_study.ipynb`](examples/radiosonde_LCL_case_study.ipynb)
- **Status:** Complete!