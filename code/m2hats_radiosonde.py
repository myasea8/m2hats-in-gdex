"""
M2HATS ISS1 RS41 radiosonde NetCDF -> corrected NetCDF for GDEX.

The soundings stay one NetCDF file per ascent or descent (see
``notebooks/archive/ISS_radiosonde.ipynb`` for why no Zarr store is made).
Each file is rewritten as NetCDF-4 (zlib-compressed) with two changes:

* Descent timing. In the source descent files, ``time``, ``launch_time`` and
  ``reference_time`` hold seconds since launch but their units are referenced
  to balloon burst, so every descent timestamp is late by the length of the
  ascent. The reference time is replaced with the launch time of the matching
  ascent; the stored values are unchanged. The ``BalloonRelease*`` global
  attributes are corrected the same way.
* Variable names and units follow the ISFS Zarr stores (``m2hats_to_zarr.py``):
  e.g. ``tdry`` -> ``air_temperature``, ``u_wind`` -> ``wind_u``, units
  ``degC`` -> ``degree_C``, ``m/s`` -> ``m s-1``. The source name and units
  are kept in ``source_short_name`` / ``source_units``.

Typical use
-----------
    from m2hats_radiosonde import convert_all
    convert_all(source_dir, output_dir)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

# Source name -> (output name, units, standard_name or None). Names match the
# ISFS stores where the quantity exists there (air_pressure, air_temperature,
# relative_humidity, wind_u/v/w, wind_speed, wind_from_direction).
RS_VAR_METADATA: dict[str, tuple[str, str, str | None]] = {
    "pres":    ("air_pressure",                     "hPa",           "air_pressure"),
    "tdry":    ("air_temperature",                  "degree_C",      "air_temperature"),
    "dp":      ("dew_point_temperature",            "degree_C",      "dew_point_temperature"),
    "rh":      ("relative_humidity",                "percent",       "relative_humidity"),
    "u_wind":  ("wind_u",                           "m s-1",         "eastward_wind"),
    "v_wind":  ("wind_v",                           "m s-1",         "northward_wind"),
    "w_wind":  ("wind_w",                           "m s-1",         "upward_air_velocity"),
    "wspd":    ("wind_speed",                       "m s-1",         "wind_speed"),
    "wdir":    ("wind_from_direction",              "degree",        "wind_from_direction"),
    "dz":      ("sonde_vertical_velocity",          "m s-1",         None),
    "mr":      ("humidity_mixing_ratio",            "g kg-1",        "humidity_mixing_ratio"),
    "vt":      ("virtual_temperature",              "K",             "virtual_temperature"),
    "theta":   ("potential_temperature",            "K",             "air_potential_temperature"),
    "theta_e": ("equivalent_potential_temperature", "K",             "equivalent_potential_temperature"),
    "theta_v": ("virtual_potential_temperature",    "K",             None),
    "alt":     ("altitude",                         "m",             "altitude"),
    "gpsalt":  ("gps_altitude",                     "m",             "altitude"),
    "lat":     ("latitude",                         "degrees_north", "latitude"),
    "lon":     ("longitude",                        "degrees_east",  "longitude"),
}
# Surface reference observation at launch: reference_<source> -> reference_<output>.
RS_VAR_METADATA.update({
    f"reference_{src}": (f"reference_{out}", units, None)
    for src, (out, units, _) in list(RS_VAR_METADATA.items())
    if src in {"pres", "tdry", "rh", "wspd", "wdir", "lat", "lon", "alt"}
})

TIME_VARS = ("time", "launch_time", "reference_time")


def _launch_reference(asc_path: Path) -> pd.Timestamp:
    """Launch time of an ascent file (the reference time of its units)."""
    with xr.open_dataset(asc_path) as asc:
        return pd.Timestamp(asc.launch_time.values)


def fix_descent_time(ds: xr.Dataset, launch: pd.Timestamp) -> xr.Dataset:
    """Re-reference a descent's time variables from burst to `launch`.

    `ds` must be opened with ``decode_times=False`` so the stored seconds are
    kept as-is; only the reference time in the units changes.
    """
    ds = ds.copy()
    units = f"seconds since {launch:%Y-%m-%d %H:%M:%S} UTC"
    for v in TIME_VARS:
        ds[v].attrs["units"] = units
    ds.attrs["BalloonReleaseDateAndTime"] = f"{launch:%Y-%m-%dT%H:%M:%S}"
    ds.attrs["BalloonReleaseTime"] = f"{launch:%H:%M:%S}"
    ds.attrs["descent_time_correction"] = (
        "Source time units were referenced to balloon burst although the values "
        "are seconds since launch; units re-referenced to the launch time of the "
        "matching ascent file. Stored values are unchanged.")
    return ds


def standardize_names(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables and set units/standard_names to match the ISFS stores."""
    ds = ds.rename({k: v[0] for k, v in RS_VAR_METADATA.items() if k in ds.variables})
    for src, (out, units, std_name) in RS_VAR_METADATA.items():
        if out not in ds.variables:
            continue
        attrs = ds[out].attrs
        attrs["source_short_name"] = src
        if attrs.get("units") != units:
            attrs["source_units"] = attrs.get("units")
        attrs["units"] = units
        if std_name is not None:
            attrs["standard_name"] = std_name
    # The source has these axes swapped (lat=X, lon=Y).
    ds["latitude"].attrs["axis"] = "Y"
    ds["longitude"].attrs["axis"] = "X"
    return ds


def convert_file(src: Path, out_dir: Path) -> Path:
    """Write the corrected, renamed copy of one sounding file to `out_dir`."""
    ds = xr.open_dataset(src, decode_times=False)
    if src.stem.endswith("_dsc"):
        ds = fix_descent_time(ds, _launch_reference(src.with_name(src.name.replace("_dsc", "_asc"))))
    ds = standardize_names(ds)
    ds.attrs["history"] = (f"{ds.attrs.get('history', '')}\n" if ds.attrs.get("history") else "") + (
        "Variables renamed to match the M2HATS ISFS Zarr stores (source names in "
        "source_short_name) by m2hats_radiosonde.py"
        + ("; descent time reference corrected." if src.stem.endswith("_dsc") else "."))
    out = out_dir / src.name
    # NetCDF-4 (the source is NetCDF-3 Classic) with lossless zlib compression.
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.variables
                if ds[v].dtype.kind in "fi" and ds[v].ndim > 0}
    ds.load().to_netcdf(out, format="NETCDF4", encoding=encoding)
    ds.close()
    return out


def convert_all(source_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Convert every sounding in `source_dir` into `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [convert_file(f, output_dir) for f in sorted(Path(source_dir).glob("*.nc"))]
