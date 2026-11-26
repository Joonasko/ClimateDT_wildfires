r"""Run FWI for one day and conditionally launch all nine WISE scenarios.

The regular daily FWI product is checked at the grid cell nearest each WISE
ignition location. If any value is at least 21.3, WISE is launched for the
same date, 08:00-20:00 local time, using the previous day's FWI components.
The noon-based FWI trigger makes this a retrospective rather than a forecast
of the start of that day's simulation.

Example (inside the WISE container):
    python -m wise.run_wildfires --year 2022 --month 6 --day 6 \
        --input-dir /input_data --output-dir /fwi_output

The weather input directory must be writable and contain the hourly inputs
needed by both applications; FWI may also need adjacent-day weather. Persist
the FWI output directory across daily runs, in chronological order. WISE
output must be mounted at /wise_output. Use --help for the inherited FWI
options, including --extra-input-dir, --initialize and --restart-file.

The daily FWI file and valid values at every ignition area are required.
For a fresh integration, FWI normally skips its first requested day; provide
adequate initialization weather or a suitable restart before relying on that
day's output. WISE retains its existing behavior for previous-day components:
missing files produce a warning and template defaults; invalid components
can cause individual areas to be skipped. WISE reports scenario failures in
its status output, so a launch does not guarantee nine successful results.
"""

from argparse import RawDescriptionHelpFormatter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from fwi.run_fwi import existing_daily_product, run_fwi, scenario_output_paths
from fwi.run_wildfires_fwi import _get_parser
from wise.run_wise import AREA_COORDS, run_wise


FWI_THRESHOLD = 21.3


def _nearest_coordinate(values, target, name, cyclic=False):
    """Return the nearest grid index, rejecting invalid or uncovered locations.

    Longitude is cyclic so that, for example, -8 and 352 degrees coincide.
    For a partial longitude grid, the large gap between its endpoints is
    treated as outside the grid, allowing half a grid spacing at each edge.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"Invalid FWI {name} coordinates")
    if name == "lat" and np.any(np.abs(values) > 90):
        raise ValueError("Invalid FWI latitude coordinates")
    if name == "lon" and np.any((values < -180) | (values > 360)):
        raise ValueError("Invalid FWI longitude coordinates")

    coordinates = values % 360 if cyclic else values
    ordered = np.sort(coordinates)
    if np.any(np.diff(ordered) <= 0):
        raise ValueError(f"Duplicate FWI {name} coordinates")
    point = target % 360 if cyclic else target
    if len(ordered) == 1:
        distance = abs((ordered[0] - point + 180) % 360 - 180) if cyclic else abs(ordered[0] - point)
        if distance > 1e-6:
            raise ValueError(f"Ignition {name} {target} is outside the FWI grid")
    elif cyclic:
        gaps = np.diff(np.append(ordered, ordered[0] + 360))
        largest = int(np.argmax(gaps))
        other_gaps = np.delete(gaps, largest)
        if gaps[largest] > 1.5 * np.median(other_gaps):
            start = ordered[(largest + 1) % len(ordered)]
            offset = (point - start) % 360
            width = 360 - gaps[largest]
            margin = np.median(other_gaps) / 2
            if width + margin < offset < 360 - margin:
                raise ValueError(f"Ignition {name} {target} is outside the FWI grid")
    else:
        if point < ordered[0] - (ordered[1] - ordered[0]) / 2 or point > ordered[-1] + (ordered[-1] - ordered[-2]) / 2:
            raise ValueError(f"Ignition {name} {target} is outside the FWI grid")
    distance = np.abs((coordinates - point + 180) % 360 - 180) if cyclic else np.abs(coordinates - point)
    return int(np.argmin(distance))


def threshold_met(fwi_file, simulation_date):
    """Check all areas against the threshold in a regular daily FWI file.

    Require one matching date, one-dimensional geographic coordinates, and a
    finite nonnegative FWI value at each location. A missing or invalid value
    raises an error rather than silently treating that area as below threshold.
    """
    with xr.open_dataset(fwi_file) as dataset:
        for name in ("time", "lat", "lon"):
            if name not in dataset.coords or dataset[name].dims != (name,):
                raise ValueError(f"{fwi_file}: expected one-dimensional {name} coordinates")
        if "fwi" not in dataset or set(dataset.fwi.dims) != {"time", "lat", "lon"}:
            raise ValueError(f"{fwi_file}: expected fwi(time, lat, lon)")
        dates = dataset.time.dt.strftime("%Y-%m-%d").values
        matching = np.flatnonzero(dates == simulation_date.isoformat())
        if len(matching) != 1:
            raise ValueError(f"{fwi_file}: expected exactly one FWI time for {simulation_date}")

        values = []
        for index, (lat, lon) in enumerate(AREA_COORDS, start=1):
            lat_index = _nearest_coordinate(dataset.lat.values, lat, "lat")
            lon_index = _nearest_coordinate(dataset.lon.values, lon, "lon", cyclic=True)
            value = dataset.fwi.isel(time=int(matching[0]), lat=lat_index, lon=lon_index).values
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{fwi_file}: invalid FWI value {value} for area{index}")
            print(f"area{index}: FWI={float(value):.4f}", flush=True)
            values.append(value)
        values = np.asarray(values)
        # FWI output is float32: compare its stored representation of 21.3.
        threshold_dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
        threshold = np.asarray(FWI_THRESHOLD, dtype=threshold_dtype)
        return bool(np.any(values >= threshold))


def run_wildfires(args):
    """Run one FWI day, then launch WISE if any area reaches the threshold.

    Args are parsed using FWI's command-line parser plus --wise-output-dir.
    Monthly chunks, preparation-only runs and disabled component output are
    incompatible with this daily runner. Historical FWI uses the output root;
    projection output and its historical boundary use FWI's own routing.
    On the first projection day, WISE receives the previous component file
    from the historical boundary rather than the projection directory.
    """
    if args.day is None:
        raise ValueError("--day is required for a combined daily run")
    if args.monthly_chunk or args.prepare_only or not args.write_components:
        raise ValueError("Combined runs require daily FWI with component output enabled")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for persistent FWI output")

    simulation_date = date(args.year, args.month, args.day)
    run_fwi(args)
    fwi_output, boundary = scenario_output_paths(args, [simulation_date], Path(args.output_dir))
    fwi_file = existing_daily_product(fwi_output, simulation_date)
    if not fwi_file.is_file():
        raise FileNotFoundError(
            f"Daily FWI output missing for {simulation_date}: {fwi_file}. "
            "The first date of a fresh FWI integration is normally skipped."
        )
    if not threshold_met(fwi_file, simulation_date):
        print(f"FWI below {FWI_THRESHOLD} at all areas; WISE not launched.", flush=True)
        return

    wise_output = Path(args.wise_output_dir)
    if wise_output.resolve() != Path("/wise_output").resolve():
        raise ValueError("WISE output must be mounted at /wise_output inside the container")
    wise_output.mkdir(parents=True, exist_ok=True)
    print(f"FWI threshold {FWI_THRESHOLD} met; launching all WISE scenarios.", flush=True)
    year, month, day = simulation_date.strftime("%Y %m %d").split()
    start_marker = fwi_output / "FWI_projection_start_date.txt"
    previous_components = None
    if boundary is not None and start_marker.exists():
        if date.fromisoformat(start_marker.read_text().strip()) == simulation_date:
            previous_date = simulation_date - timedelta(days=1)
            previous_components = boundary / f"{previous_date:%Y_%m_%d}_fwi_comp.nc"
    fwi_input = (
        {"fwi_file": str(previous_components)}
        if previous_components is not None else {"fwi_dir": str(fwi_output)}
    )
    run_wise(
        str(args.hpctmpdir), str(wise_output),
        year, month, day, year, month, day,
        **fwi_input, scenario_start_hour=8, scenario_end_hour=20,
    )


def main(argv=None):
    """Parse FWI options and the WISE output mount, then run one day."""
    parser = _get_parser()
    parser.description = __doc__
    parser.formatter_class = RawDescriptionHelpFormatter
    parser.add_argument(
        "--wise-output-dir", default="/wise_output",
        help="WISE output directory, mounted at /wise_output in the container",
    )
    run_wildfires(parser.parse_args(argv))


if __name__ == "__main__":
    main()