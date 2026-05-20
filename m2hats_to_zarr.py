"""
M2HATS high-rate netCDF -> Zarr restructuring.

Converts the flat ISFS variable-name encoding (var_height_site) of the M2HATS
high-rate product into a CF-compliant xarray DataTree with physical dimensions,
sub-second sample offsets, validity flags, and Blosc-Zstd-compressed Zarr
output.

Typical use
-----------
    import xarray as xr
    from m2hats_to_zarr import restructure_m2hats

    ds = xr.open_mfdataset("isfs_m2hats_hr_qc_geo_tiltcor_hr_2023*.nc",
                           combine="by_coords")          # your existing step
    restructure_m2hats(ds, output_path="m2hats.zarr",
                      tilt_corrected=True)

Output layout (DataTree, written as Zarr groups)
------------------------------------------------
    m2hats.zarr/
    |-- array/                          # 50-tower horizontal array, ~4 m
    |   |-- sonic_60hz                  (time, sample,    site_a)
    |   |-- sonic_50hz                  (time, sample_50, site_b)
    |   |-- sonic_30hz                  (time, sample_30, site_c)
    |   |-- irga_60hz                   (time, sample,    site_irga)
    |   |-- barometer_20hz              (time, sample_20, site_irga)
    |   `-- trh_1hz                     (time,            site_irga)
    `-- profile_t0/                     # multi-level tower at t0
        |-- sonic_60hz                  (time, sample,    height)
        |-- irga_60hz                   (time, sample,    height)
        |-- barometer_20hz              (time, sample_20, height)
        `-- trh_1hz                     (time,            height)

Conventions
-----------
* All variables get CF `standard_name`, `units`, `long_name`, `_FillValue=NaN`.
* Each high-rate group carries a `sample_offset` coord (seconds within the
  outer 1-Hz time bin) so that the true observation time is
  `time + sample_offset`.
* Each group dimensioned on `time` carries a `valid` flag variable using
  CF flag_values/flag_meanings, encoding the pre-31-July t0 invalidity and
  the five t0 tower-lowering windows from Table 4 of the data report.
* Source fill value 1.e+37 is converted to NaN before write.

Caveats
-------
* This is a structural rewrite; it does NOT recompute any physical quantities
  (winds remain whatever frame the input is in; the `tilt_corrected` flag is
  recorded in attrs but no rotation is performed).
* Author has not run this on the actual ~250 GB store. Test on a one-day
  subset first and inspect the resulting tree with `xr.open_datatree(...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

try:
    from numcodecs import Blosc
    _COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
except ImportError:           # numcodecs is bundled with zarr; fallback to None
    _COMPRESSOR = None


# ===========================================================================
# Static metadata baked from the M2HATS Flux Data Report (v1.1, 2024-08-30)
# ===========================================================================

# Table 1 (A): horizontal array site locations and surveyed heights.
# Layout: site -> (height_m, longitude_deg, latitude_deg).
# The t0p..t18 Leica reference correction (-1.311e-5 deg lon, -6.92e-6 deg lat)
# is already applied in the report; we use those corrected values verbatim.
ARRAY_SITES: dict[str, tuple[float, float, float]] = {
    "t0p": (4.484, -117.0742621, 38.04264114),
    "t1":  (4.524, -117.0742073, 38.04264879),
    "t2":  (4.539, -117.0741494, 38.04265703),
    "t3":  (4.562, -117.0740934, 38.04266465),
    "t4":  (4.637, -117.0740379, 38.04267236),
    "t5":  (4.505, -117.0739823, 38.04268031),
    "t6":  (4.577, -117.0739246, 38.04268885),
    "t7":  (4.476, -117.0738686, 38.04269536),
    "t8":  (4.477, -117.0738127, 38.04270436),
    "t9":  (4.435, -117.0737576, 38.04271223),
    "t10": (4.435, -117.0737003, 38.04272029),
    "t11": (4.382, -117.0736437, 38.04272789),
    "t12": (4.445, -117.0735887, 38.04273585),
    "t13": (4.368, -117.0735322, 38.04274399),
    "t14": (4.275, -117.0734764, 38.04275169),
    "t15": (4.268, -117.0734201, 38.04275979),
    "t16": (4.25,  -117.0733642, 38.04276774),
    "t17": (4.18,  -117.0733096, 38.0427752),
    "t18": (4.307, -117.0732521, 38.04278321),
    "t19": (4.241, -117.073196,  38.04279101),
    "t20": (4.196, -117.0731418, 38.04279866),
    "t21": (4.223, -117.0730852, 38.04280654),
    "t22": (4.234, -117.0730274, 38.0428141),
    "t23": (4.19,  -117.0729731, 38.04282168),
    "t24": (4.241, -117.0729152, 38.04282979),
    "t25": (4.192, -117.0728595, 38.04283755),
    "t26": (4.19,  -117.0728028, 38.04284525),
    "t27": (4.262, -117.0727484, 38.04285281),
    "t28": (4.236, -117.0726928, 38.0428604),
    "t29": (4.21,  -117.072635,  38.04286808),
    "t30": (4.344, -117.0725804, 38.04287555),
    "t31": (4.34,  -117.0725238, 38.0428834),
    "t32": (4.326, -117.0724658, 38.04288916),
    "t33": (4.417, -117.0724114, 38.04289897),
    "t34": (4.393, -117.0723555, 38.04290635),
    "t35": (4.399, -117.0723000, 38.04291432),
    "t36": (4.424, -117.0722415, 38.04292203),
    "t37": (4.421, -117.0721880, 38.04292989),
    "t38": (4.303, -117.0721322, 38.04293735),
    "t39": (4.384, -117.0720752, 38.04294517),
    "t40": (4.391, -117.0720169, 38.04295332),
    "t41": (4.347, -117.0719620, 38.04296114),
    "t42": (4.416, -117.0719060, 38.04296853),
    "t43": (4.435, -117.0718510, 38.04297646),
    "t44": (4.413, -117.0717935, 38.04298414),
    "t45": (4.439, -117.0717377, 38.04299194),
    "t46": (4.422, -117.0716806, 38.04299969),
    "t47": (4.356, -117.0716247, 38.04300773),
    "t48": (4.416, -117.0715677, 38.04301529),
    "t49": (4.406, -117.0715126, 38.04302335),
}

# Table 1 (B): t0 multi-level tower, nominal_label_m -> (actual_height_m, lon, lat)
T0_PROFILE: dict[float, tuple[float, float, float]] = {
    0.5: (0.615,  -117.0704707, 38.0431674),
    1.0: (1.166,  -117.0704705, 38.04316741),
    2.0: (2.114,  -117.0704705, 38.04316738),
    3.0: (3.016,  -117.0704704, 38.04316738),
    4.0: (4.198,  -117.0704701, 38.04316737),
    7.0: (6.894,  -117.0699262, 38.04373068),
    15.0: (15.453, -117.0699341, 38.0437454),
    28.0: (28.551, -117.0699382, 38.04374603),
}

# Appendix A: CSAT model at each array tower. We split into a "configuration"
# string so users can filter (e.g. ds.where(ds.csat_model.str.contains("EC150"))).
# t1 swapped CSAT3 -> CSAT3A on 2023-08-08 13:00 UTC; we record the late
# configuration as primary and note the swap in `csat_model_note`.
CSAT_MODEL: dict[str, str] = {
    "t0p": "CSAT3A",        "t1":  "CSAT3A",      "t2":  "CSAT3A+EC150",
    "t3":  "CSAT3",         "t4":  "CSAT3",       "t5":  "CSAT3A+EC150",
    "t6":  "CSAT3B",        "t7":  "CSAT3",       "t8":  "CSAT3A+EC150",
    "t9":  "CSAT3A",        "t10": "CSAT3",       "t11": "CSAT3A+EC150",
    "t12": "CSAT3B",        "t13": "CSAT3",       "t14": "CSAT3A+EC150",
    "t15": "CSAT3A",        "t16": "CSAT3",       "t17": "CSAT3A+EC150",
    "t18": "CSAT3B",        "t19": "CSAT3",       "t20": "CSAT3A+EC150",
    "t21": "CSAT3A",        "t22": "CSAT3",       "t23": "CSAT3A+EC150",
    "t24": "CSAT3B",        "t25": "CSAT3",       "t26": "CSAT3A+EC150",
    "t27": "CSAT3A",        "t28": "CSAT3",       "t29": "CSAT3A+EC150",
    "t30": "CSAT3B",        "t31": "CSAT3",       "t32": "CSAT3A+EC150",
    "t33": "CSAT3",         "t34": "CSAT3",       "t35": "CSAT3A+EC150",
    "t36": "CSAT3B",        "t37": "CSAT3",       "t38": "CSAT3A+EC150",
    "t39": "CSAT3",         "t40": "CSAT3",       "t41": "CSAT3A+EC150",
    "t42": "CSAT3B",        "t43": "CSAT3",       "t44": "CSAT3A+EC150",
    "t45": "CSAT3",         "t46": "CSAT3",       "t47": "CSAT3A+EC150",
    "t48": "CSAT3B",        "t49": "CSAT3",
}

# In the high-rate netCDF, sonic data at each array site is recorded at the
# rate that matches its CSAT model. Map sample dim -> rate, and the inverse
# `CSAT_RATE_DIM` mapping kept for reference. Routing decisions use the
# OBSERVED source dim, not the Appendix A table, because the table doesn't
# document every mid-campaign instrument swap (e.g., t7 in 20230807_07 has
# sonic data on `sample_50`, implying an undocumented CSAT3B deployment).
SAMPLE_DIM_TO_RATE: dict[str, int] = {
    "sample": 60, "sample_50": 50, "sample_30": 30, "sample_20": 20,
}
CSAT_RATE_DIM: dict[str, str] = {
    "CSAT3":         "sample_30",
    "CSAT3B":        "sample_50",
    "CSAT3A":        "sample",
    "CSAT3A+EC150":  "sample",
}

# Periods to flag as invalid for t0 (UTC). PDT -> UTC = +7 h.
# Source: Table 4 (tower lowerings) + p. 8 (early relocation).
T0_PROFILE_VALID_START = np.datetime64("2023-07-31T00:00:00")
T0_LOWERING_WINDOWS: list[tuple[str, str, int, str]] = [
    # (start_utc, end_utc, flag_value, name)
    ("2023-07-24T17:30", "2023-07-24T18:30", 2, "first_lowering"),
    ("2023-07-25T17:00", "2023-07-25T23:00", 2, "first_relocation"),
    ("2023-07-31T19:30", "2023-08-01T00:30", 2, "second_lowering"),
    ("2023-09-03T14:45", "2023-09-03T17:20", 3, "battery_swap"),
    ("2023-09-04T19:00", "2023-09-04T20:30", 2, "third_lowering"),
]

# Variable -> (out_name, units, standard_name_or_None, long_name)
VAR_METADATA: dict[str, tuple[str, str, str | None, str]] = {
    # CSAT 3D sonic
    "u":        ("wind_u",            "m s-1",    "eastward_wind",
                 "Sonic wind U component"),
    "v":        ("wind_v",            "m s-1",    "northward_wind",
                 "Sonic wind V component"),
    "w":        ("wind_w",            "m s-1",    "upward_air_velocity",
                 "Sonic wind W component"),
    "spd":      ("wind_speed",        "m s-1",    "wind_speed",
                 "Horizontal wind speed"),
    "dir":      ("wind_from_direction","degree",  "wind_from_direction",
                 "Wind from direction"),
    "tc":       ("sonic_temperature", "degree_C", None,
                 "Virtual air temperature from sonic speed of sound"),
    "ldiag":    ("sonic_qc_flag",     "1",        "status_flag",
                 "CSAT3 logical diagnostic: 0=OK, 1=any diagbit set"),
    "diagbits": ("sonic_diag_bits",   "1",        "status_flag",
                 "CSAT3 diag bit sum (1=lo sig, 2=hi sig, 4=no lock, "
                 "8=path diff, 16=skipped samp)"),
    "diag":     ("sonic_diag_bits",   "1",        "status_flag",
                 "CSAT3B diagnostic sum"),
    # EC150 open-path IRGA
    "h2o":      ("water_vapor_density","g m-3",
                 "mass_concentration_of_water_vapor_in_air",
                 "Water vapor density"),
    "co2":      ("co2_density",       "g m-3",
                 "mass_concentration_of_carbon_dioxide_in_air",
                 "CO2 density"),
    "Pirga":    ("irga_pressure",     "hPa",      "air_pressure",
                 "EC150 IRGA cell pressure"),
    "Tirga":    ("irga_temperature",  "degree_C", "air_temperature",
                 "EC150 IRGA cell temperature"),
    "irgadiag": ("irga_qc_flag",      "1",        "status_flag",
                 "EC150 IRGA diagnostic (0=OK)"),
    # SHT85 hygro-thermometer
    "T":        ("air_temperature",   "degree_C", "air_temperature",
                 "Air temperature (Sensirion SHT85 hygro-thermometer)"),
    "RH":       ("relative_humidity", "percent",  "relative_humidity",
                 "Relative humidity (Sensirion SHT85)"),
    # Paroscientific 6000 nanobarometer
    "P":        ("air_pressure",      "hPa",      "air_pressure",
                 "Barometric pressure (Paroscientific 6000 nanobarometer)"),
}

# Variable -> instrument-class group key
SONIC_VARS  = {"u", "v", "w", "tc", "ldiag", "diagbits", "diag"}
IRGA_VARS   = {"h2o", "co2", "Pirga", "Tirga", "irgadiag"}
TRH_VARS    = {"T", "RH"}
BARO_VARS   = {"P"}

# Source fill value used by NIDAS-produced ISFS netCDF files.
NIDAS_FILL_VALUE = 1.0e37


# ===========================================================================
# Variable-name parser
# ===========================================================================

_VAR_RE = re.compile(
    r"""^
    (?P<var>.+?)                # variable token (lazy; may contain '_')
    _+                           # 1 or 2 underscores ('__' marks a moment)
    (?P<hnum>\d+(?:_\d+)?)       # height value, '_' acts as decimal point
    (?P<hunit>cm|m)              # unit
    _(?P<site>t\d+p?)            # site: t0, t0p, t1..t49
    $""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class ParsedName:
    var: str          # ISFS short variable name, e.g. "u", "Pirga"
    height_m: float   # positive = above ground, negative = below (soil)
    site: str         # "t0", "t0p", "t1"..."t49"
    is_moment: bool   # True if name used '__' separator (2nd/3rd moment)


def parse_var(name: str) -> ParsedName | None:
    """Parse `<var>_<height-spec>_<site>`. Returns None if it doesn't match.

    Examples
    --------
    >>> parse_var("u_4m_t17")
    ParsedName(var='u', height_m=4.0, site='t17', is_moment=False)
    >>> parse_var("Pirga_0_5m_t0")
    ParsedName(var='Pirga', height_m=0.5, site='t0', is_moment=False)
    >>> parse_var("u_w__7m_t0")
    ParsedName(var='u_w', height_m=7.0, site='t0', is_moment=True)
    >>> parse_var("Tsoil_3_1cm_t23")
    ParsedName(var='Tsoil', height_m=-0.031, site='t23', is_moment=False)
    """
    m = _VAR_RE.match(name)
    if not m:
        return None
    # If two-or-more underscores precede the height, it's a moment variable.
    # Detect by looking for "__" in the original between var token and height.
    between = name[len(m["var"]): name.index(m["hnum"] + m["hunit"] + "_" + m["site"])]
    is_moment = "__" in between
    h = float(m["hnum"].replace("_", "."))
    if m["hunit"] == "cm":
        h = -h / 100.0
    return ParsedName(var=m["var"], height_m=h, site=m["site"], is_moment=is_moment)


# ===========================================================================
# Routing: (parsed name) -> output group path
# ===========================================================================

def classify(p: ParsedName, source_dims: tuple[str, ...]) -> str | None:
    """Return the output group path for a parsed variable, or None to skip.

    Sonic variables are routed by their OBSERVED source sample dim rather
    than by the Appendix A CSAT model table, which doesn't track every
    mid-campaign instrument swap.

    Skips: unknown var (moments, GPS diagnostics, etc.); unknown site; or
    a sonic var whose source dim isn't one of the recognized sample dims.
    """
    if p.is_moment:
        return None
    if p.var not in VAR_METADATA:
        return None

    if p.site == "t0":
        sub = ("sonic_60hz"     if p.var in SONIC_VARS  else
               "irga_60hz"      if p.var in IRGA_VARS   else
               "trh_1hz"        if p.var in TRH_VARS    else
               "barometer_20hz" if p.var in BARO_VARS   else None)
        return f"profile_t0/{sub}" if sub else None

    if p.site not in CSAT_MODEL:
        return None

    if p.var in SONIC_VARS:
        # Route by source dim. e.g., sample_50 -> sonic_50hz, sample -> sonic_60hz.
        for d in source_dims:
            if d in SAMPLE_DIM_TO_RATE and d != "sample_20":
                return f"array/sonic_{SAMPLE_DIM_TO_RATE[d]}hz"
        return None
    if p.var in IRGA_VARS:
        return "array/irga_60hz"
    if p.var in TRH_VARS:
        return "array/trh_1hz"
    if p.var in BARO_VARS:
        return "array/barometer_20hz"
    return None


# ===========================================================================
# Group construction
# ===========================================================================

def _site_sort_key(site: str) -> tuple:
    """Sort 't0p' before 't1', then numerically."""
    if site == "t0p":
        return (0, -1)
    if site == "t0":
        return (-1, 0)
    return (1, int(site[1:]))


def _sample_offset_da(sample_dim: str, rate_hz: int) -> xr.DataArray:
    """Centered sub-second offset: time + offset = true observation time.

    The source `time` axis is the CENTER of each 1-second bin (verified on
    file 20230807_07: timestamps end in `.500000000`). Within each bin,
    `rate_hz` samples are uniformly distributed across [time-0.5, time+0.5).
    Offsets are sub-bin centers minus the bin center, so:

        true_time = time + sample_offset

    For 60 Hz: offsets run from -0.4917 s to +0.4917 s in steps of 1/60.
    """
    offsets = (np.arange(rate_hz, dtype="float64") + 0.5) / rate_hz - 0.5
    return xr.DataArray(
        offsets,
        dims=(sample_dim,),
        attrs={
            "units": "s",
            "long_name": "Offset from 1-Hz time-bin center",
            "description": ("Add to `time` to get the true observation "
                            "timestamp. The `time` axis is the bin center; "
                            "this offset gives the centered position of "
                            "each sub-sample within the bin."),
        },
    )


def _build_validity_flag(
    time: xr.DataArray,
    is_profile_t0: bool,
) -> xr.DataArray:
    """Construct a per-time integer flag using CF flag_values / flag_meanings.

    Flag values
    -----------
    0  ok
    1  t0 pre-31-July (towers were being repositioned; profiles unreliable)
    2  t0 30 m trailer tower lowered for maintenance
    3  t0 battery swap

    For non-t0 groups, only value 0 is ever set; the variable is still
    written so downstream code can branch uniformly.
    """
    flag = np.zeros(time.size, dtype="int8")
    if is_profile_t0:
        t = time.values.astype("datetime64[ns]")
        flag = np.where(t < T0_PROFILE_VALID_START, 1, flag)
        for start, end, value, _name in T0_LOWERING_WINDOWS:
            mask = (t >= np.datetime64(start)) & (t < np.datetime64(end))
            flag = np.where(mask, value, flag)
    return xr.DataArray(
        flag,
        dims=("time",),
        attrs={
            "long_name": "Data validity flag",
            "standard_name": "status_flag",
            "flag_values": np.array([0, 1, 2, 3], dtype="int8"),
            "flag_meanings": "ok pre_31july_towers_moving "
                             "t0_tower_lowered t0_battery_swap",
            "comment": ("Times with flag != 0 are present but should not be "
                        "used for science. See Table 4 of the M2HATS report."),
        },
    )


def _apply_cf_metadata(
    da: xr.DataArray,
    isfs_name: str,
    *,
    extra_attrs: dict | None = None,
) -> xr.DataArray:
    out_name, units, std_name, long_name = VAR_METADATA[isfs_name]
    # Preserve the source's hand-written long_name (e.g. "Wind U component,
    # CSAT3BH") since it carries the instrument variant that our routing
    # otherwise only keeps via the `csat_model` site coord.
    source_long_name = da.attrs.get("long_name")
    attrs = {
        "long_name": long_name,
        "units": units,
        "isfs_short_name": isfs_name,
    }
    if source_long_name and source_long_name != long_name:
        attrs["isfs_long_name"] = source_long_name
    if std_name is not None:
        attrs["standard_name"] = std_name
    if extra_attrs:
        attrs.update(extra_attrs)
    da = da.rename(out_name)
    da.attrs = attrs
    # NIDAS writes _FillValue = 1.e+37, but stored as float32 the actual
    # value is ~9.99999993e+36. Mask anything above 1e36 to be robust
    # whether the source was opened with mask_and_scale=True or False.
    if np.issubdtype(da.dtype, np.floating):
        da = da.where(da < 1e36)
    return da


def _stack_array_group(
    entries: dict[str, list[tuple[float, str, xr.DataArray]]],
    sample_dim: str | None,
    instrument: str,
) -> xr.Dataset:
    """Build an `array/*` group: stack each variable along a `site` dim.

    Parameters
    ----------
    entries : {isfs_var: [(height_m, site, DataArray), ...], ...}
    sample_dim : the source dim used for this group ('sample', 'sample_50',
        'sample_30', 'sample_20'), or None for 1 Hz groups.
    instrument : human-readable instrument label for group attrs.
    """
    # Union of sites that have any variable in this group.
    sites = sorted({s for v in entries.values() for _, s, _ in v},
                   key=_site_sort_key)
    out_vars: dict[str, xr.DataArray] = {}
    for isfs_var, items in entries.items():
        per_site = {s: da for _, s, da in items}
        # Build a template from one of the existing arrays so missing sites
        # get NaN slabs of the right shape & dtype.
        template = next(iter(per_site.values()))
        slabs: list[xr.DataArray] = []
        for s in sites:
            if s in per_site:
                slabs.append(per_site[s])
            else:
                empty = xr.full_like(template, np.nan, dtype="float32")
                slabs.append(empty)
        stacked = xr.concat(
            slabs,
            dim=pd.Index(sites, name="site"),
            coords="minimal",
            compat="override",
        )
        out_vars[VAR_METADATA[isfs_var][0]] = _apply_cf_metadata(stacked, isfs_var)

    ds = xr.Dataset(out_vars)

    # site-level coordinates
    ds = ds.assign_coords(
        site_height=("site", [ARRAY_SITES[s][0] for s in sites]),
        site_lon=("site",    [ARRAY_SITES[s][1] for s in sites]),
        site_lat=("site",    [ARRAY_SITES[s][2] for s in sites]),
        csat_model_appendix_a=("site",
                               [CSAT_MODEL.get(s, "unknown") for s in sites]),
    )
    ds["site_height"].attrs = {"long_name": "Surveyed sensor height above ground",
                               "units": "m"}
    ds["site_lon"].attrs    = {"long_name": "Site longitude (Leica-corrected)",
                               "units": "degrees_east", "standard_name": "longitude"}
    ds["site_lat"].attrs    = {"long_name": "Site latitude (Leica-corrected)",
                               "units": "degrees_north", "standard_name": "latitude"}
    ds["csat_model_appendix_a"].attrs = {
        "long_name": "CSAT 3D sonic configuration per Appendix A of the data report",
        "comment": ("Static reference from Appendix A. Does NOT always match the "
                    "actual deployed instrument: e.g. t7 in the 2023-08-07 file "
                    "carries sonic data on sample_50 (CSAT3B-rate) despite being "
                    "listed as CSAT3 here. The authoritative configuration is the "
                    "group this site appears in (sonic_30hz / 50hz / 60hz)."),
    }

    if sample_dim is not None:
        rate = {"sample": 60, "sample_50": 50, "sample_30": 30, "sample_20": 20}[sample_dim]
        ds = ds.assign_coords(sample_offset=_sample_offset_da(sample_dim, rate))

    ds["valid"] = _build_validity_flag(ds["time"], is_profile_t0=False)
    ds.attrs.update({
        "subarray": "horizontal_50_tower",
        "instrument_class": instrument,
        "Conventions": "CF-1.10",
    })
    return ds


def _stack_profile_group(
    entries: dict[str, list[tuple[float, str, xr.DataArray]]],
    sample_dim: str | None,
    instrument: str,
) -> xr.Dataset:
    """Build a `profile_t0/*` group: stack each variable along a `height` dim."""
    heights = sorted({h for v in entries.values() for h, _, _ in v})
    out_vars: dict[str, xr.DataArray] = {}
    for isfs_var, items in entries.items():
        per_h = {h: da for h, s, da in items if s == "t0"}
        template = next(iter(per_h.values()))
        slabs: list[xr.DataArray] = []
        for h in heights:
            if h in per_h:
                slabs.append(per_h[h])
            else:
                slabs.append(xr.full_like(template, np.nan, dtype="float32"))
        stacked = xr.concat(
            slabs,
            dim=pd.Index(heights, name="height"),
            coords="minimal",
            compat="override",
        )
        out_vars[VAR_METADATA[isfs_var][0]] = _apply_cf_metadata(stacked, isfs_var)

    ds = xr.Dataset(out_vars)
    actual = [T0_PROFILE[h][0] for h in heights]
    ds = ds.assign_coords(
        height_actual=("height", actual),
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

    if sample_dim is not None:
        rate = {"sample": 60, "sample_50": 50, "sample_30": 30, "sample_20": 20}[sample_dim]
        ds = ds.assign_coords(sample_offset=_sample_offset_da(sample_dim, rate))

    ds["valid"] = _build_validity_flag(ds["time"], is_profile_t0=True)
    ds.attrs.update({
        "subarray": "t0_multi_level_profile",
        "instrument_class": instrument,
        "Conventions": "CF-1.10",
    })
    return ds


# ===========================================================================
# Encoding (chunking + compression) per group
# ===========================================================================

def _encoding_for(ds: xr.Dataset, time_chunk: int) -> dict:
    """One day's worth of time per chunk by default. Sample/site/height are
    un-chunked. Float vars get NaN as _FillValue; ints get the source fill.
    """
    enc = {}
    for name, da in ds.data_vars.items():
        chunks = []
        for d in da.dims:
            if d == "time":
                chunks.append(min(time_chunk, ds.sizes["time"]))
            else:
                chunks.append(ds.sizes[d])
        e = {"chunks": tuple(chunks)}
        if _COMPRESSOR is not None:
            e["compressor"] = _COMPRESSOR
        if np.issubdtype(da.dtype, np.floating):
            e["dtype"] = "float32"
            e["_FillValue"] = np.float32("nan")
        enc[name] = e
    return enc


# ===========================================================================
# Public entry point
# ===========================================================================

def restructure_m2hats(
    ds: xr.Dataset,
    output_path: str,
    *,
    tilt_corrected: bool = True,
    time_chunk_seconds: int = 3600,
    write: bool = True,
) -> xr.DataTree:
    """Restructure the flat M2HATS high-rate dataset into a Zarr DataTree.

    Parameters
    ----------
    ds : xarray.Dataset
        Your existing combined high-rate dataset. Expected dims:
        time, sample, sample_50, sample_30, sample_20.
    output_path : str
        Path for the output `.zarr` store.
    tilt_corrected : bool, default True
        Recorded in attrs only. Pass False if your source comes from the
        non-tiltcor netCDFs.
    time_chunk_seconds : int, default 3600
        Zarr chunk size along time, in seconds (= rows because time is 1 Hz).
    write : bool, default True
        If False, return the DataTree without writing (handy for inspection).

    Returns
    -------
    xarray.DataTree
        The constructed tree. Already written to disk if `write=True`.
    """
    # Group source variables by destination.
    # buckets[group_path][isfs_var] -> list of (height_m, site, DataArray)
    buckets: dict[str, dict[str, list[tuple[float, str, xr.DataArray]]]] = {}
    skipped: list[str] = []
    for name in ds.data_vars:
        p = parse_var(name)
        if p is None:
            skipped.append(name); continue
        dest = classify(p, ds[name].dims)
        if dest is None:
            skipped.append(name); continue
        bucket = buckets.setdefault(dest, {}).setdefault(p.var, [])
        bucket.append((p.height_m, p.site, ds[name]))

    # Build each group.
    group_specs = {
        "array/sonic_60hz":     ("sample",    "CSAT3A 3D sonic anemometer (60 Hz)"),
        "array/sonic_50hz":     ("sample_50", "CSAT3B 3D sonic anemometer (50 Hz)"),
        "array/sonic_30hz":     ("sample_30", "CSAT3 3D sonic anemometer (30 Hz)"),
        "array/irga_60hz":      ("sample",    "EC150 open-path IRGA (60 Hz)"),
        "array/barometer_20hz": ("sample_20", "Paroscientific 6000 nanobarometer (20 Hz)"),
        "array/trh_1hz":        (None,        "Sensirion SHT85 hygro-thermometer (1 Hz)"),
        "profile_t0/sonic_60hz":     ("sample",    "CSAT3A 3D sonic anemometer (60 Hz)"),
        "profile_t0/irga_60hz":      ("sample",    "EC150 open-path IRGA (60 Hz)"),
        "profile_t0/barometer_20hz": ("sample_20", "Paroscientific 6000 nanobarometer (20 Hz)"),
        "profile_t0/trh_1hz":        (None,        "Sensirion SHT85 hygro-thermometer (1 Hz)"),
    }

    nodes: dict[str, xr.Dataset] = {}
    for group_path, (sample_dim, instrument) in group_specs.items():
        entries = buckets.get(group_path)
        if not entries:
            continue
        if group_path.startswith("array/"):
            nodes[group_path] = _stack_array_group(entries, sample_dim, instrument)
        else:
            nodes[group_path] = _stack_profile_group(entries, sample_dim, instrument)

    # Top-level attrs. Propagate source globals under `source_*` for traceability.
    root_attrs = {
        "title": "M2HATS ISFS Surface Meteorology and Flux Products (restructured)",
        "summary": ("NCAR/EOL ISFS high-rate surface flux measurements from the "
                    "M2HATS campaign, restructured from flat ISFS variable naming "
                    "(var_height_site) into a CF-compliant DataTree."),
        "source_dataset_doi": "10.26023/HW9Z-MF0D-NX04",
        "campaign": "M2HATS",
        "location": "Tonopah, Nevada, USA",
        "time_coverage_start": "2023-07-23",
        "time_coverage_end":   "2023-09-24",
        "tilt_corrected": str(tilt_corrected),
        "time_axis_convention": "bin_center_1Hz",
        "Conventions": "CF-1.10",
        "history": ("Restructured from ISFS high-rate netCDF3 (NIDAS v1.2.1-8 "
                    "output) into Zarr by m2hats_to_zarr.py."),
        "skipped_source_vars": ",".join(sorted(skipped)) if skipped else "",
    }
    for k, v in ds.attrs.items():
        root_attrs[f"source_{k}"] = v
    root = xr.Dataset(attrs=root_attrs)

    tree = xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in nodes.items()}})

    if write:
        encoding = {}
        for path, node in nodes.items():
            for vname, e in _encoding_for(node, time_chunk_seconds).items():
                encoding[f"/{path}/{vname}"] = e
        tree.to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)
    return tree


# ===========================================================================
# Minimal smoke test for the parser (run as `python m2hats_to_zarr.py`)
# ===========================================================================
if __name__ == "__main__":
    cases = {
        "u_4m_t17":         ("u", 4.0, "t17", False),
        "Pirga_4m_t17":     ("Pirga", 4.0, "t17", False),
        "T_4m_t14":         ("T", 4.0, "t14", False),
        "h2o_28m_t0":       ("h2o", 28.0, "t0", False),
        "P_4m_t2":          ("P", 4.0, "t2", False),
        "u_0_5m_t0":        ("u", 0.5, "t0", False),
        "Tsoil_3_1cm_t23":  ("Tsoil", -0.031, "t23", False),
        "u_w__7m_t0":       ("u_w", 7.0, "t0", True),
        "w_w_tc__0_5m_t0":  ("w_w_tc", 0.5, "t0", True),
        "u_4m_t0p":         ("u", 4.0, "t0p", False),
    }
    for src, expected in cases.items():
        got = parse_var(src)
        assert got is not None, f"parse failed for {src!r}"
        assert (got.var, got.height_m, got.site, got.is_moment) == expected, \
            f"{src!r}: got {got}, expected {expected}"
        print(f"OK  {src:24s} -> {got}")
    print("All parser smoke tests passed.")