"""
M2HATS 5-minute-averaged netCDF -> Zarr restructuring.

Converts the flat ISFS variable-name encoding (``<var>_<height>_<site>`` for
scalars, ``<var1>_<var2>__<height>_<site>`` for 2nd/3rd moments) of the
M2HATS 5-minute geo-rotated, tilt-corrected product into a CF-compliant
xarray DataTree with physical dimensions, validity flags, and
Blosc-Zstd-compressed Zarr output.

This module deliberately follows the conventions of ``m2hats_to_zarr.py``
(the high-rate workflow) and imports its site tables, name parser, validity
flag, and encoding helpers so that both stores read identically:

* Variables are renamed via ``VAR_METADATA`` (e.g. ``u`` -> ``wind_u``,
  ``h2o`` -> ``water_vapor_density``) and carry CF ``units``,
  ``standard_name``, ``long_name`` plus ``isfs_short_name`` /
  ``isfs_long_name`` for traceability.
* Site ``t0`` is the multi-level profile tower and appears only under
  ``/profile_t0``; the horizontal array uses ``t0p``..``t49`` (``t0p`` data
  first appears in the 5-minute files on 2023-07-28).
* The t1 sonic swap (CSAT3 -> CSAT3A at 2023-08-23T13:00Z) is split into
  site ``t1p`` (pre-swap) and ``t1`` (post-swap), each NaN-padded outside its
  valid period -- see ``fix_t1``.
* t0 down times are NOT masked. Every group carries a ``valid`` flag variable
  (CF flag_values/flag_meanings) built from ``T0_PROFILE_VALID_START`` and
  ``T0_LOWERING_WINDOWS``, exactly as in the high-rate store.
* The source fill value (~1e37) is converted to NaN at restructure time.

Unlike the high-rate product, 5-minute files have no ``sample`` dimension
(each variable is already one value per time step) and timestamps are
absolute ``datetime64`` bin centres, so no ``base_time`` reconstruction or
``sample_offset`` coordinate is needed.

Typical use
-----------
    import xarray as xr
    from m2hats_5min_to_zarr import fix_t1, fill_missing_vars, restructure_5min

    ds_day = fill_missing_vars(fix_t1(xr.open_dataset(nc_file)), templates)
    ds_day.to_zarr(working_zarr, mode="w", consolidated=True)   # or append

    ds = xr.open_zarr(working_zarr)
    tree = restructure_5min(ds, output_path="m2hats_5min.zarr")

Output layout (DataTree, written as Zarr groups)
------------------------------------------------
    m2hats_5min.zarr/
    |-- array/                    # 50-tower horizontal array, ~4 m
    |   |-- sonic                 (site, time)         + 2nd/3rd moments
    |   |-- irga                  (site, time)         + 2nd moments
    |   |-- trh                   (site, time)
    |   `-- barometer             (site, time)
    |-- profile_t0/               # multi-level tower at t0
    |   |-- sonic                 (height, time)       + 2nd/3rd moments
    |   |-- irga                  (height, time)       + 2nd moments
    |   |-- trh                   (height, time)
    |   `-- barometer             (height, time)
    |-- soils/
    |   |-- heat_flux             (site, time)
    |   |-- temperature           (site, depth, time)
    |   |-- moisture              (site, time)
    |   `-- thermal_props         (site, time)
    `-- radiation/
        `-- nr01                  (site, time)

Moment naming
-------------
The high-rate product has no moments, so this module extends the naming
strategy: 2nd moments become ``cov_<a>_<b>`` and 3rd moments
``m3_<a>_<b>_<c>``, where ``a``, ``b``, ``c`` are the ISFS component tokens,
lower-cased (``u, v, w, tc, h2o, co2, p``; e.g. ISFS ``w_P`` ->
``cov_w_p``). The original name is kept in ``isfs_short_name``.

Caveats
-------
* This is a structural rewrite; it does not recompute any physical
  quantities. ``tilt_corrected`` is recorded in attrs only.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from m2hats_to_zarr import (
    ARRAY_SITES,
    CSAT_MODEL,
    T0_PROFILE,
    T0_PROFILE_VALID_START,     # re-exported for the notebook
    T0_LOWERING_WINDOWS,        # re-exported for the notebook
    VAR_METADATA as _HR_VAR_METADATA,
    SONIC_VARS as _HR_SONIC_VARS,
    IRGA_VARS as _HR_IRGA_VARS,
    TRH_VARS as _HR_TRH_VARS,
    BARO_VARS,
    parse_var,
    resolve_tilt_corrected,
    time_attrs,
    wind_frame_text,
    _site_sort_key,
    _build_validity_flag,
    _encoding_for,
)


# ===========================================================================
# Static metadata
# ===========================================================================

# t1 sonic swap: CSAT3 (30 Hz) -> CSAT3A (60 Hz). Same instant as the
# high-rate `fix_time_and_t1` split.
T1_SWAP_UTC = np.datetime64("2023-08-23T13:00:00")

# Variable -> (out_name, units, standard_name_or_None, long_name).
# Starts from the high-rate table and adds the variables that only exist in
# the 5-minute product.
VAR_METADATA: dict[str, tuple[str, str, str | None, str]] = {
    **_HR_VAR_METADATA,
    # EC150 open-path IRGA
    "SSh2o":     ("irga_h2o_signal_strength", "1", None,
                  "EC150 IRGA H2O signal strength"),
    "SSco2":     ("irga_co2_signal_strength", "1", None,
                  "EC150 IRGA CO2 signal strength"),
    # NCAR hygrothermometer
    "Rfan":      ("trh_fan_speed",       "min-1",    None,
                  "Aspiration fan speed (NCAR hygrothermometer)"),
    # Soils
    "Gsoil":     ("soil_heat_flux",      "W m-2",    "downward_heat_flux_in_soil",
                  "Soil heat flux"),
    "Qsoil":     ("soil_moisture",       "m3 m-3",
                  "volume_fraction_of_condensed_water_in_soil",
                  "Volumetric soil moisture"),
    "Tsoil":     ("soil_temperature",    "degree_C", "soil_temperature",
                  "Soil temperature"),
    "Lambda":    ("soil_thermal_conductivity", "W m-1 K-1", None,
                  "Soil thermal conductivity (thermal property probe)"),
    "Tau63":     ("soil_probe_decay_time_constant", "s", None,
                  "Thermal property probe decay time constant"),
    "Vheat":     ("soil_probe_heater_voltage", "V", None,
                  "Thermal property probe heater voltage"),
    "Vpile_On":  ("soil_probe_pile_voltage_on", "V", None,
                  "Thermal property probe pile voltage, heater on"),
    "Vpile_Off": ("soil_probe_pile_voltage_off", "V", None,
                  "Thermal property probe pile voltage, heater off"),
    # Hukseflux NR01 radiometer + Decagon leaf wetness
    "Rsw_in":    ("shortwave_in",        "W m-2",
                  "surface_downwelling_shortwave_flux_in_air",
                  "Incoming shortwave radiation (Hukseflux NR01)"),
    "Rsw_out":   ("shortwave_out",       "W m-2",
                  "surface_upwelling_shortwave_flux_in_air",
                  "Outgoing shortwave radiation (Hukseflux NR01)"),
    "Rpile_in":  ("thermopile_in",       "W m-2",    None,
                  "Incoming longwave thermopile signal (Hukseflux NR01)"),
    "Rpile_out": ("thermopile_out",      "W m-2",    None,
                  "Outgoing longwave thermopile signal (Hukseflux NR01)"),
    "Tcase":     ("radiometer_case_temperature", "degree_C", None,
                  "Radiometer case temperature (Hukseflux NR01)"),
    "Wetness":   ("leaf_wetness_voltage", "V",       None,
                  "Leaf wetness sensor output (Decagon)"),
}

# Variable -> instrument-class group key (high-rate sets + 5-min extras)
SONIC_VARS = _HR_SONIC_VARS | {"spd", "dir"}
IRGA_VARS  = _HR_IRGA_VARS  | {"SSh2o", "SSco2"}
TRH_VARS   = _HR_TRH_VARS   | {"Rfan"}

# Moment component tokens; a moment goes to irga if it involves a gas or
# pressure, otherwise to sonic.
_MOMENT_TOKENS = {"u", "v", "w", "tc", "h2o", "co2", "P"}
_IRGA_MOMENT_TOKENS = {"h2o", "co2", "P"}

# Soil / radiation variables carry no height token: `<var>_<site>`.
SOIL_GROUPS = {
    "Gsoil":     "soils/heat_flux",
    "Qsoil":     "soils/moisture",
    "Tsoil":     "soils/temperature",
    "Lambda":    "soils/thermal_props",
    "Tau63":     "soils/thermal_props",
    "Vheat":     "soils/thermal_props",
    "Vpile_On":  "soils/thermal_props",
    "Vpile_Off": "soils/thermal_props",
}
RAD_VARS = {"Rsw_in", "Rsw_out", "Rpile_in", "Rpile_out", "Tcase", "Wetness"}
_NOHEIGHT_RE = re.compile(
    r"^(?P<var>" + "|".join(sorted(set(SOIL_GROUPS) | RAD_VARS, key=len, reverse=True))
    + r")_(?P<site>t\d+)$"
)

# The data report's co2 bias corrections, which apply to this product.
_CO2_BIAS_NOTE = (
    "co2 at sites t0 (1 m), t0 (4 m), t20 (4 m), and t47 (4 m) had low "
    "biases corrected: +0.168, +0.110, +0.098, +0.086 g/m^3 respectively. "
    "co2 at t0 28 m removed through 2023-09-04 13:00 PDT."
)


# ===========================================================================
# Source file preprocessing
# ===========================================================================

_T1_SUFFIX_RE = re.compile(r"_t1$")


def fix_t1(ds: xr.Dataset) -> xr.Dataset:
    """Split site t1 at the sonic swap into t1p (pre) and t1 (post).

    Mirrors the high-rate ``fix_time_and_t1`` convention: before
    2023-08-23T13:00Z site t1 held a CSAT3 and its data is labelled ``t1p``;
    after, it held a CSAT3A and its data stays ``t1``. Both columns exist in
    every file and are NaN outside their valid period, so every day
    concatenates cleanly. 5-minute bins are labelled at their centre, so a
    bin is pre-swap iff its centre is before the swap time.
    """
    t1_vars = [v for v in ds.data_vars if _T1_SUFFIX_RE.search(v) and "time" in ds[v].dims]
    if not t1_vars:
        return ds
    pre = ds["time"] < T1_SWAP_UTC
    fixed = {}
    for v in t1_vars:
        fixed[_T1_SUFFIX_RE.sub("_t1p", v)] = ds[v].where(pre)
        fixed[v] = ds[v].where(~pre)
    return ds.assign(fixed)


def fill_missing_vars(ds: xr.Dataset, templates: dict) -> xr.Dataset:
    """NaN-fill any variable in `templates` that's absent from `ds`.

    Not every variable appears in every daily file (e.g. a sensor was
    added, removed, or renamed partway through the campaign). If a
    variable is simply absent from a day's file, appending that file to
    the combined Zarr store leaves that variable's array shorter than the
    others -- which is what produces the "conflicting sizes for dimension
    'time'" error on readback. `templates` should map each variable name
    across the *whole* campaign to its `dims`, `other_dims`, and `dtype` so
    that gaps can be padded with correctly shaped, correctly typed NaNs.
    """
    n_time = ds.sizes["time"]
    missing = {}
    for v, spec in templates.items():
        if v in ds:
            continue
        shape = tuple(
            n_time if d == "time" else spec["other_dims"][d]
            for d in spec["dims"]
        )
        missing[v] = xr.DataArray(
            np.full(shape, np.nan, dtype=spec["dtype"]),
            dims=spec["dims"],
            coords={"time": ds["time"]},
        )
    return ds.assign(missing) if missing else ds


# ===========================================================================
# Metadata
# ===========================================================================

def _metadata(isfs_name: str, is_moment: bool) -> tuple[str, str, str | None, str] | None:
    """Return (out_name, units, standard_name, long_name) or None if unknown."""
    if not is_moment:
        return VAR_METADATA.get(isfs_name)
    comps = isfs_name.split("_")
    if len(comps) not in (2, 3) or not all(c in _MOMENT_TOKENS for c in comps):
        return None
    comp_meta = [VAR_METADATA[c] for c in comps]
    names = [m[0] for m in comp_meta]
    units = " ".join(m[1] for m in comp_meta)
    tokens = "_".join(c.lower() for c in comps)
    if len(comps) == 2:
        return (f"cov_{tokens}", units, None,
                f"Covariance of {names[0]} and {names[1]}")
    return (f"m3_{tokens}", units, None,
            f"Third-order moment of {names[0]}, {names[1]} and {names[2]}")


def _apply_cf_metadata(da: xr.DataArray, isfs_name: str, meta: tuple) -> xr.DataArray:
    """Same behaviour as ``m2hats_to_zarr._apply_cf_metadata`` with a 5-min table."""
    out_name, units, std_name, long_name = meta
    source_long_name = da.attrs.get("long_name")
    attrs = {
        "long_name": long_name,
        "units": units,
        "isfs_short_name": isfs_name,
    }
    if source_long_name and source_long_name != long_name:
        attrs["isfs_long_name"] = source_long_name
    if da.attrs.get("units"):
        attrs["isfs_units"] = da.attrs["units"]
    if std_name is not None:
        attrs["standard_name"] = std_name
    da = da.rename(out_name)
    da.attrs = attrs
    # NIDAS writes _FillValue = 1.e+37 (~9.99999993e+36 as float32).
    if np.issubdtype(da.dtype, np.floating):
        da = da.where(da < 1e36)
    return da


# ===========================================================================
# Routing: flat name -> (group path, isfs_var, key, is_moment)
# ===========================================================================

def classify(name: str):
    """Return (group_path, isfs_var, key, is_moment) or None to skip.

    `key` is the stacking label: site for array/soils/radiation groups,
    nominal height (m) for profile_t0 groups, and (depth_m, site) for soil
    temperature.
    """
    m = _NOHEIGHT_RE.match(name)
    if m:
        var, site = m["var"], m["site"]
        grp = SOIL_GROUPS.get(var, "radiation/nr01")
        return grp, var, site, False

    p = parse_var(name)
    if p is None:
        return None

    if p.var == "Tsoil":
        return "soils/temperature", "Tsoil", (-p.height_m, p.site), False

    if _metadata(p.var, p.is_moment) is None:
        return None

    if p.is_moment:
        comps = set(p.var.split("_"))
        inst = "irga" if comps & _IRGA_MOMENT_TOKENS else "sonic"
    else:
        inst = ("sonic"     if p.var in SONIC_VARS else
                "irga"      if p.var in IRGA_VARS  else
                "trh"       if p.var in TRH_VARS   else
                "barometer" if p.var in BARO_VARS  else None)
        if inst is None:
            return None

    if p.site == "t0":
        if p.height_m not in T0_PROFILE:
            return None
        return f"profile_t0/{inst}", p.var, p.height_m, p.is_moment
    if p.site in ARRAY_SITES:
        return f"array/{inst}", p.var, p.site, p.is_moment
    return None


# ===========================================================================
# Group construction
# ===========================================================================

def _stack(entries: dict, dim: str, labels: list) -> dict[str, xr.DataArray]:
    """Stack each variable's per-label DataArrays along `dim`, NaN-padding gaps."""
    out = {}
    for (isfs_var, is_moment), per_label in entries.items():
        template = next(iter(per_label.values()))
        slabs = [per_label[k] if k in per_label
                 else xr.full_like(template, np.nan, dtype="float32")
                 for k in labels]
        stacked = xr.concat(slabs, dim=pd.Index(labels, name=dim),
                            coords="minimal", compat="override")
        meta = _metadata(isfs_var, is_moment)
        out[meta[0]] = _apply_cf_metadata(stacked, isfs_var, meta)
    return out


def _assign_site_coords(ds: xr.Dataset, sites: list, with_csat: bool) -> xr.Dataset:
    ds = ds.assign_coords(
        site_height=("site", [ARRAY_SITES[s][0] for s in sites]),
        site_lon=("site",    [ARRAY_SITES[s][1] for s in sites]),
        site_lat=("site",    [ARRAY_SITES[s][2] for s in sites]),
    )
    ds["site_height"].attrs = {"long_name": "Surveyed sensor height above ground",
                               "units": "m"}
    ds["site_lon"].attrs    = {"long_name": "Site longitude (Leica-corrected)",
                               "units": "degrees_east", "standard_name": "longitude"}
    ds["site_lat"].attrs    = {"long_name": "Site latitude (Leica-corrected)",
                               "units": "degrees_north", "standard_name": "latitude"}
    if with_csat:
        ds = ds.assign_coords(csat_model_appendix_a=(
            "site", [CSAT_MODEL.get(s, "unknown") for s in sites]))
        ds["csat_model_appendix_a"].attrs = {
            "long_name": "CSAT 3D sonic configuration per Appendix A of the data report",
            "comment": ("Static reference from Appendix A. Does NOT always match the "
                        "actual deployed instrument (see the high-rate store, where "
                        "the sonic_30hz / 50hz / 60hz group is authoritative)."),
        }
    return ds


def _build_array_group(entries: dict, instrument: str, subarray: str,
                       with_csat: bool) -> xr.Dataset:
    sites = sorted({s for per in entries.values() for s in per}, key=_site_sort_key)
    ds = xr.Dataset(_stack(entries, "site", sites))
    ds = _assign_site_coords(ds, sites, with_csat)
    ds["valid"] = _build_validity_flag(ds["time"], is_profile_t0=False)
    ds.attrs.update({
        "subarray": subarray,
        "instrument_class": instrument,
        "Conventions": "CF-1.10",
    })
    return ds


def _build_profile_group(entries: dict, instrument: str) -> xr.Dataset:
    heights = sorted({h for per in entries.values() for h in per})
    ds = xr.Dataset(_stack(entries, "height", heights))
    ds = ds.assign_coords(
        height_actual=("height", [T0_PROFILE[h][0] for h in heights]),
        height_lon=("height", [T0_PROFILE[h][1] for h in heights]),
        height_lat=("height", [T0_PROFILE[h][2] for h in heights]),
    )
    ds["height"].attrs = {"long_name": "Nominal sensor height above ground",
                          "units": "m", "axis": "Z", "positive": "up"}
    ds["height_actual"].attrs = {"long_name": "Surveyed sensor height above ground",
                                 "units": "m"}
    ds["height_lon"].attrs = {"long_name": "Sensor longitude", "units": "degrees_east",
                              "standard_name": "longitude"}
    ds["height_lat"].attrs = {"long_name": "Sensor latitude", "units": "degrees_north",
                              "standard_name": "latitude"}
    ds["valid"] = _build_validity_flag(ds["time"], is_profile_t0=True)
    ds.attrs.update({
        "subarray": "t0_multi_level_profile",
        "instrument_class": instrument,
        "Conventions": "CF-1.10",
    })
    return ds


def _build_soil_temperature_group(entries: dict) -> xr.Dataset:
    per_key = entries[("Tsoil", False)]
    sites = sorted({s for _, s in per_key}, key=_site_sort_key)
    depths = sorted({d for d, _ in per_key})
    template = next(iter(per_key.values()))
    by_site = []
    for s in sites:
        slabs = [per_key.get((d, s), xr.full_like(template, np.nan, dtype="float32"))
                 for d in depths]
        by_site.append(xr.concat(slabs, dim=pd.Index(depths, name="depth"),
                                 coords="minimal", compat="override"))
    stacked = xr.concat(by_site, dim=pd.Index(sites, name="site"),
                        coords="minimal", compat="override")
    ds = xr.Dataset({VAR_METADATA["Tsoil"][0]:
                     _apply_cf_metadata(stacked, "Tsoil", VAR_METADATA["Tsoil"])})
    ds["depth"].attrs = {"long_name": "Sensor depth below ground", "units": "m",
                         "axis": "Z", "positive": "down"}
    ds = _assign_site_coords(ds, sites, with_csat=False)
    ds["valid"] = _build_validity_flag(ds["time"], is_profile_t0=False)
    ds.attrs.update({
        "subarray": "soils",
        "instrument_class": "Soil temperature probes",
        "Conventions": "CF-1.10",
    })
    return ds


# ===========================================================================
# Public entry point
# ===========================================================================

GROUP_SPECS: dict[str, str] = {
    "array/sonic":           "CSAT3 / CSAT3A / CSAT3B 3D sonic anemometers (5-min statistics)",
    "array/irga":            "EC150 open-path IRGA (5-min statistics)",
    "array/trh":             "NCAR hygrothermometer, Sensirion SHT85 (5-min means)",
    "array/barometer":       "Paroscientific 6000 nanobarometer (5-min means)",
    "profile_t0/sonic":      "CSAT3A 3D sonic anemometer (5-min statistics)",
    "profile_t0/irga":       "EC150 open-path IRGA (5-min statistics)",
    "profile_t0/trh":        "NCAR hygrothermometer, Sensirion SHT85 (5-min means)",
    "profile_t0/barometer":  "Paroscientific 6000 nanobarometer (5-min means)",
    "soils/heat_flux":       "Soil heat flux plates (5-min means)",
    "soils/temperature":     "Soil temperature probes (5-min means)",
    "soils/moisture":        "Soil moisture probes (5-min means)",
    "soils/thermal_props":   "Soil thermal property probes (5-min means)",
    "radiation/nr01":        "Hukseflux NR01 radiometer and Decagon leaf wetness (5-min means)",
}


def restructure_5min(
    ds: xr.Dataset,
    output_path: str,
    *,
    tilt_corrected: bool | None = None,
    time_chunk: int = 2016,      # ~1 week of 5-min data
    write: bool = True,
) -> xr.DataTree:
    """Restructure the flat M2HATS 5-minute dataset into a Zarr DataTree.

    Parameters
    ----------
    ds : xarray.Dataset
        The combined flat 5-minute dataset (after ``fix_t1``).
    output_path : str
        Path for the output `.zarr` store.
    tilt_corrected : bool, optional
        Recorded in attrs and the title only. Taken from the source
        `wind3d_tilt_correction` attr when present; a value that contradicts
        it raises ValueError.
    time_chunk : int, default 2016
        Zarr chunk size along time, in 5-minute steps.
    write : bool, default True
        If False, return the DataTree without writing (handy for inspection).

    Returns
    -------
    xarray.DataTree
        The constructed tree. Already written to disk if `write=True`.
    """
    # buckets[group_path][(isfs_var, is_moment)][key] -> DataArray
    buckets: dict[str, dict[tuple[str, bool], dict]] = {}
    skipped: list[str] = []
    for name in ds.data_vars:
        c = classify(name) if "time" in ds[name].dims else None
        if c is None:
            skipped.append(name); continue
        dest, var, key, is_moment = c
        buckets.setdefault(dest, {}).setdefault((var, is_moment), {})[key] = ds[name]

    nodes: dict[str, xr.Dataset] = {}
    for group_path, instrument in GROUP_SPECS.items():
        entries = buckets.get(group_path)
        if not entries:
            continue
        if group_path == "soils/temperature":
            nodes[group_path] = _build_soil_temperature_group(entries)
        elif group_path.startswith("profile_t0/"):
            nodes[group_path] = _build_profile_group(entries, instrument)
        elif group_path.startswith("array/"):
            nodes[group_path] = _build_array_group(
                entries, instrument, "horizontal_50_tower", with_csat=True)
        else:
            nodes[group_path] = _build_array_group(
                entries, instrument, group_path.split("/")[0], with_csat=False)
        nodes[group_path]["time"].attrs = {**nodes[group_path]["time"].attrs,
                                           **time_attrs(300)}

    # Top-level attrs. Propagate source globals under `source_*` for traceability.
    tilt_corrected = resolve_tilt_corrected(ds, tilt_corrected)
    title_suffix, wind_sentence = wind_frame_text(tilt_corrected)
    root_attrs = {
        "title": ("M2HATS ISFS 5-Minute Surface Meteorology and Flux Products, "
                  f"{title_suffix} (restructured)"),
        "summary": ("NCAR/EOL ISFS 5-minute surface flux statistics from the "
                    "M2HATS campaign, restructured from flat ISFS variable naming "
                    f"(var_height_site) into a CF-compliant DataTree. {wind_sentence}"),
        "campaign": "M2HATS",
        "location": "Tonopah, Nevada, USA",
        "time_coverage_start": "2023-07-23",
        "time_coverage_end":   "2023-09-24",
        "tilt_corrected": str(tilt_corrected),
        "time_axis_convention": "bin_center_5min",
        "Conventions": "CF-1.10",
        "history": ("Restructured from ISFS 5-minute netCDF (NIDAS v1.2.1-8 "
                    "output) into Zarr by m2hats_5min_to_zarr.py."),
        "co2_bias_correction_note": _CO2_BIAS_NOTE,
        "skipped_source_vars": ",".join(sorted(skipped)) if skipped else "",
    }
    for k, v in ds.attrs.items():
        root_attrs[f"source_{k}"] = v
    root = xr.Dataset(attrs=root_attrs)

    tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in nodes.items()}})

    if write:
        # Daily source files give 288-step Dask chunks; align them with the
        # Zarr time chunk so parallel writes never share a chunk.
        # Non-time dims are written whole, matching `_encoding_for`.
        nodes = {k: v.chunk({**{d: -1 for d in v.dims if d != "time"},
                             "time": time_chunk})
                 for k, v in nodes.items()}
        tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in nodes.items()}})
        # DataTree.to_zarr takes encoding keyed by group path, then variable.
        encoding = {f"/{path}": _encoding_for(node, time_chunk)
                    for path, node in nodes.items()}
        tree.to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)
    return tree


# ===========================================================================
# Misc utilities
# ===========================================================================

def dir_size_mb(path: str) -> float:
    """Recursively sum file sizes under `path`, return MB."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6
