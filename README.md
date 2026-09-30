# Hosting EOL datasets from M2HATS in GDEX
This repository outlines the process for standardizing the datasets collected during the Multi-point Monin-Obukhov similarity horizontal array turbulence study (M2HATS) and hosted in the EOL FDA. The intent of the standardization is to increase the accessability of the data, which will also be improved by its placement into CISL's Geoscience Data Exchange (GDEX). 

### About M2HATS
The M2HATS field campaign was conducted in the summer of 2023 in Tonopah, NV, and utilized many of ISF's instruments. M2HATS is used as a pilot project for collaboration with GDEX because it is our most recent field campaign and contains data structures that are most likely to appear in future campaigns.

### Data storage considerations
Each dataset is evaluated separately: converting to a new format is only worthwhile when it makes the data meaningfully easier to use. The per-dataset details and status are in [datasets.md](datasets.md).

#### Choosing a format
We ask four questions of each dataset:

1. **How large is it, and how is it read?** Large datasets that users read in slices (one site, one variable, one day) benefit most from a chunked, cloud-friendly format such as Zarr. Small datasets can be read whole in seconds from any format.
2. **How fragmented is it?** Thousands of hourly files, each holding every variable, force users to open many files to extract one time series. Consolidating them into one store removes that step.
3. **What is its natural structure?** Regular time series across many sites map onto multidimensional arrays (`site × time × sample`). Ragged, event-based data such as balloon soundings (each a different length) fit one file per event, and forcing them into a shared array means padding.
4. **Is there an established community format, and what metadata would a conversion lose?** Radiosonde users and tools expect one CF trajectory file per sounding, with per-sounding metadata in global attributes that a combined store cannot keep.

#### Decisions so far

| Dataset | Source (FDA) | GDEX | Why |
|---|---|---|---|
| ISFS high-rate, geographic and tilt-corrected | ~250 GB each, ~1,500 hourly NetCDF-3 files | One Zarr v3 store each (~197 GB) | Large, highly fragmented, and read in slices. Restructured from flat `var_height_site` names into `site`/`height` dimensions. |
| ISFS 5-minute | 140 MB, daily NetCDF-3 files | One Zarr v3 store (112 MB) | Small, but the same restructuring as the high-rate data gives both products an identical layout and naming. |
| ISS radiosondes | 58 MB, 237 NetCDF-3 files | 237 NetCDF-4 files (57 MB) | Small, ragged, already in the standard format. A Zarr only saved a few seconds of load time and dropped most per-sounding metadata ([evaluation](notebooks/archive/ISS_radiosonde.ipynb)). |

#### Conventions shared across formats
Whatever the format, the GDEX copies follow the same conventions so that datasets can be used together:
- **Variable names and units** follow CF and are shared across datasets (e.g. `air_temperature`, `wind_u`, `air_pressure`, units such as `degree_C` and `m s-1`). The original name and units are kept in each variable's attributes (`isfs_short_name`, `source_short_name`, `source_units`).
- **Time** is UTC. For averaged data, timestamps mark the centre of each averaging bin, and `time` carries CF `standard_name`, `long_name` and `axis` attributes.
- **Quality information is flagged, not masked.** Known bad periods (e.g. t0 tower maintenance) are recorded in a `valid` flag variable, so users decide what to exclude.
- **Titles and attributes describe the processing** (for example, whether sonic winds are tilt corrected), so users can tell otherwise similar products apart.

#### Provenance and corrections
- **The FDA source files are never modified.** GDEX copies are written separately, and every change is recorded in the file's `history` attribute and in the conversion code under [`code/`](code/).
- **Metadata fixes to existing Zarr stores are made in place** (attributes, dimension names), without rewriting data, after backing up the metadata.
- **Errors found in the source data are corrected in the GDEX copy only where the fix is unambiguous, and reported to EOL.** So far these are:
  - The radiosonde descent timestamps are late by the length of each ascent. This is corrected in the GDEX copy.
  - In the 5-minute ISFS product, site t1 from 2023-07-28 21:05 to 2023-07-29 21:50 UTC appears to come from a different 60 Hz sonic. This is not corrected; it is documented only.

#### Zarr details
- **Structure:** Zarr v3 stores, one per product, organized as an xarray DataTree with one group per instrument (e.g. `/array/sonic_60hz`).
- **Chunking:** the high-rate stores use 1-hour time chunks containing all sites and sub-second samples, compressed with Blosc-Zstd. This suits reading one day or one site over many days.
- **Metadata:** consolidated, so a store opens with a single metadata read. Consolidated metadata is not yet part of the Zarr v3 specification, so non-Python readers may need to read the per-group metadata instead.
