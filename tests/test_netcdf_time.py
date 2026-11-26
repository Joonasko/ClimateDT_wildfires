import contextlib
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "src/wise/run_wise.py"
SPEC = importlib.util.spec_from_file_location("run_wise", SCRIPT)
wise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wise)

RASTERS = (
    "burn_grid", "crown_fuel_consumed", "max_crown_fraction_burned",
    "max_flame_length", "max_intensity", "surface_fuel_consumed",
    "total_fuel_consumption",
)


class NetCDFTimeTests(unittest.TestCase):
    def check_outputs(self, start, end, skipped=(), fallback=False, fwi_directory=False):
        with tempfile.TemporaryDirectory(prefix="wise-time-test-") as temporary:
            root = Path(temporary)
            inputs = root / "input"
            output = root / "output"
            inputs.mkdir()
            output.mkdir()
            for variable in ("2t", "2d", "10u", "10v", "avg_tprate"):
                if variable == "avg_tprate":
                    name = (
                        f"{start}_T00_00_to_{end}_T23_00_"
                        "avg_tprate_raw_data.nc"
                    )
                else:
                    name = (
                        f"{start}_T00_to_{end}_T23_{variable}_"
                        "timestep_60_hourly_mean.nc"
                    )
                xr.Dataset(
                    {variable: ("time", [1.0])},
                    coords={"time": [np.datetime64(start.replace("_", "-"))]},
                ).to_netcdf(inputs / name)

            scenarios = [f"scenario_{i}" for i in range(1, 10)]
            burn = np.zeros((4, 6), dtype="uint8")
            burn[1:3, 2:4] = 1
            continuous = burn.astype("float32")
            continuous[0, 0] = -9999
            for scenario in scenarios:
                directory = output / scenario
                directory.mkdir()
                for name in RASTERS:
                    data = burn if name == "burn_grid" else continuous
                    with rasterio.open(
                        directory / f"{name}.tif", "w", driver="GTiff",
                        width=6, height=4, count=1, dtype=data.dtype,
                        crs="EPSG:4326",
                        transform=from_origin(24.15, 64.01, 0.001, 0.001),
                        nodata=None if name == "burn_grid" else -9999,
                    ) as dst:
                        dst.write(data, 1)

            result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            prepare_fwi_files = []
            def prepare(*args, **kwargs):
                prepare_fwi_files.append(kwargs["fwi_file"])
                diagnostics = kwargs["fwi_diagnostics"]
                diagnostics.update(
                    status="template_fallback" if fallback else "partial" if skipped else "valid",
                    warnings=["FWI warning"] if skipped or fallback else [],
                    areas={
                        area: {"status": "invalid_values", "error": "missing FFMC"}
                        for area in skipped
                    },
                )
                return scenarios

            with patch.object(wise, "ncdf_edits_multiarea", side_effect=prepare), \
                    patch.object(wise.subprocess, "run", return_value=result) as model, \
                    contextlib.redirect_stdout(io.StringIO()):
                wise.run_wise(
                    str(inputs), str(output),
                    *start.split("_"), *end.split("_"),
                    fwi_dir=str(root / "fwi") if fwi_directory else None,
                )
            attempted = [
                int(call.args[0][-1].split("/area")[1].split("/")[0])
                for call in model.call_args_list
            ]
            self.assertEqual(attempted, [i for i in range(1, 10) if i not in skipped])

            reports = list(output.glob("*_status.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            if fwi_directory:
                previous_date = np.datetime64(start.replace("_", "-")) - np.timedelta64(1, "D")
                expected_file = root / "fwi" / (
                    str(previous_date).replace("-", "_") + "_fwi_comp.nc"
                )
                self.assertEqual(report["fwi_file"], str(expected_file))
                self.assertEqual(report["fwi_directory"], str(root / "fwi"))
                self.assertEqual(prepare_fwi_files, [str(expected_file)])
            self.assertEqual(
                report["status"],
                "failed" if len(skipped) == 9 else "partial" if skipped else "success",
            )
            if skipped or fallback:
                self.assertEqual(report["fwi_initialization"]["warnings"], ["FWI warning"])
            if fallback:
                self.assertEqual(report["fwi_initialization"]["status"], "template_fallback")
            for area in skipped:
                self.assertEqual(report["areas"][area - 1]["status"], "fwi_skipped")
                self.assertEqual(report["areas"][area - 1]["error"], "missing FFMC")
                self.assertIsNone(report["areas"][area - 1]["return_code"])
            files = list(output.glob("scenario_*/*.nc"))
            self.assertEqual(len(files), (9 - len(skipped)) * 7)
            for area in skipped:
                self.assertFalse(list((output / f"scenario_{area}").glob("*.nc")))
            expected_date = np.datetime64(start.replace("_", "-"), "ns")
            expected_days = (
                datetime.strptime(start, "%Y_%m_%d") - datetime(1990, 1, 1)
            ).days
            for path in files:
                name = path.stem.split("_T00_00_", 1)[1]
                with self.subTest(path=path), xr.open_dataset(path) as ds:
                    self.assertEqual(ds[name].dims, ("time", "lat", "lon"))
                    self.assertEqual(ds[name].shape, (1, 4, 6))
                    self.assertEqual(ds.time.values[0], expected_date)
                    self.assertEqual(ds.time.attrs["standard_name"], "time")
                    self.assertEqual(ds.time.attrs["axis"], "T")
                    self.assertIn("daily label", ds.time.attrs["comment"])
                    self.assertEqual(ds[name].attrs["grid_mapping"], "spatial_ref")
                    self.assertEqual(ds.spatial_ref.dims, ())
                    expected = burn if name == "burn_grid" else np.where(
                        continuous == -9999, np.nan, continuous
                    )
                    np.testing.assert_allclose(
                        ds[name].isel(time=0).values, expected, rtol=1e-6, atol=1e-10
                    )
                with xr.open_dataset(path, decode_times=False) as raw:
                    self.assertEqual(raw.time.dtype, np.dtype("float64"))
                    self.assertEqual(raw.time.values[0], expected_days)
                    self.assertEqual(raw.time.attrs["units"], "days since 1990-01-01")
                    self.assertEqual(raw.time.attrs["calendar"], "standard")
                    self.assertNotIn("_FillValue", raw.time.attrs)
                with rasterio.open(f'NETCDF:"{path}":{name}') as raster:
                    self.assertEqual(raster.count, 1)
                    self.assertEqual(raster.crs.to_epsg(), 4326)
                    np.testing.assert_allclose(
                        raster.read(1), burn if name == "burn_grid" else continuous,
                        rtol=1e-6, atol=1e-10,
                    )

    def test_daily_time_coordinate(self):
        self.check_outputs("2032_08_08", "2032_08_08")

    def test_multiday_run_uses_start_date_on_leap_day(self):
        self.check_outputs("2032_02_29", "2032_03_02")

    def test_invalid_fwi_skips_only_affected_areas(self):
        self.check_outputs("2032_08_08", "2032_08_08", skipped=(2, 4))

    def test_all_invalid_fwi_areas_are_nonfatal(self):
        self.check_outputs("2032_08_08", "2032_08_08", skipped=tuple(range(1, 10)))

    def test_missing_fwi_fallback_warning_survives_success(self):
        self.check_outputs("2032_08_08", "2032_08_08", fallback=True)

    def test_directory_is_resolved_and_reported(self):
        self.check_outputs("2032_08_08", "2032_08_08", fwi_directory=True)


if __name__ == "__main__":
    unittest.main()
