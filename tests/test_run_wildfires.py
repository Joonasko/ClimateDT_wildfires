import contextlib
from datetime import date
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wise import run_wildfires


class CombinedRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.fwi_dir = self.root / "fwi"
        self.fwi_dir.mkdir()
        self.day = date(2022, 6, 6)
        self.path = self.fwi_dir / "2022_06_06_fwi.nc"
        self.lat = sorted({lat for lat, _ in run_wildfires.AREA_COORDS})
        self.lon = sorted({lon % 360 for _, lon in run_wildfires.AREA_COORDS})

    def write_fwi(self, value=10.0, high=None, timestamp="2022-06-06T12:00", dtype="float32"):
        data = np.full((1, len(self.lat), len(self.lon)), value, dtype=dtype)
        if high is not None:
            lat, lon, level = high
            data[0, self.lat.index(lat), self.lon.index(lon % 360)] = level
        xr.Dataset(
            {"fwi": (("time", "lat", "lon"), data)},
            coords={"time": [np.datetime64(timestamp)], "lat": self.lat, "lon": self.lon},
        ).to_netcdf(self.path)

    def arguments(self):
        parser = run_wildfires._get_parser()
        parser.add_argument("--wise-output-dir", default="/wise_output")
        return parser.parse_args([
            "--year", "2022", "--month", "6", "--day", "6",
            "--input-dir", str(self.input_dir), "--output-dir", str(self.fwi_dir),
        ])

    def test_threshold_uses_any_area_including_float32_boundary(self):
        for area in (0, 3):
            lat, lon = run_wildfires.AREA_COORDS[area]
            with self.subTest(area=area):
                self.write_fwi(high=(lat, lon, 21.3))
                self.assertTrue(run_wildfires.threshold_met(self.path, self.day))
        self.write_fwi(value=21.29)
        self.assertFalse(run_wildfires.threshold_met(self.path, self.day))
        self.write_fwi(value=21, dtype="int32")
        self.assertFalse(run_wildfires.threshold_met(self.path, self.day))

    def test_invalid_fwi_and_wrong_date_fail(self):
        lat, lon = run_wildfires.AREA_COORDS[0]
        self.write_fwi(high=(lat, lon, np.nan))
        with self.assertRaisesRegex(ValueError, "invalid FWI value"):
            run_wildfires.threshold_met(self.path, self.day)
        self.write_fwi(timestamp="2022-06-05T12:00")
        with self.assertRaisesRegex(ValueError, "exactly one FWI time"):
            run_wildfires.threshold_met(self.path, self.day)

    def test_uncovered_location_is_not_assigned_distant_cell(self):
        self.write_fwi()
        with patch.object(run_wildfires, "AREA_COORDS", ((0.0, 0.0),)):
            with self.assertRaisesRegex(ValueError, "outside the FWI grid"):
                run_wildfires.threshold_met(self.path, self.day)

    def test_missing_daily_output_is_error(self):
        with patch.object(run_wildfires, "run_fwi"), \
                patch.object(run_wildfires, "scenario_output_paths", return_value=(self.fwi_dir, None)), \
                patch.object(run_wildfires, "run_wise") as wise:
            with self.assertRaisesRegex(FileNotFoundError, "2022_06_06_fwi.nc"):
                run_wildfires.run_wildfires(self.arguments())
            wise.assert_not_called()

    def test_trigger_runs_wise_for_same_day_with_previous_day_directory(self):
        lat, lon = run_wildfires.AREA_COORDS[0]
        self.write_fwi(high=(lat, lon, 21.3))
        with patch.object(run_wildfires, "run_fwi") as fwi, \
                patch.object(run_wildfires, "scenario_output_paths", return_value=(self.fwi_dir, None)), \
                patch.object(run_wildfires, "run_wise") as wise, \
                patch.object(Path, "mkdir"), contextlib.redirect_stdout(io.StringIO()):
            run_wildfires.run_wildfires(self.arguments())
        fwi.assert_called_once()
        wise.assert_called_once_with(
            str(self.input_dir), "/wise_output",
            "2022", "06", "06", "2022", "06", "06",
            fwi_dir=str(self.fwi_dir), scenario_start_hour=8, scenario_end_hour=20,
        )

    def test_no_trigger_does_not_run_wise(self):
        self.write_fwi()
        with patch.object(run_wildfires, "run_fwi"), \
                patch.object(run_wildfires, "scenario_output_paths", return_value=(self.fwi_dir, None)), \
                patch.object(run_wildfires, "run_wise") as wise, \
                contextlib.redirect_stdout(io.StringIO()):
            run_wildfires.run_wildfires(self.arguments())
        wise.assert_not_called()

    def test_projection_handover_uses_boundary_previous_day(self):
        lat, lon = run_wildfires.AREA_COORDS[0]
        self.write_fwi(high=(lat, lon, 30))
        (self.fwi_dir / "FWI_projection_start_date.txt").write_text("2022-06-06\n")
        boundary = self.root / "boundary"
        with patch.object(run_wildfires, "run_fwi"), \
                patch.object(run_wildfires, "scenario_output_paths", return_value=(self.fwi_dir, boundary)), \
                patch.object(run_wildfires, "run_wise") as wise, \
                patch.object(Path, "mkdir"), contextlib.redirect_stdout(io.StringIO()):
            run_wildfires.run_wildfires(self.arguments())
        self.assertEqual(wise.call_args.kwargs["fwi_file"], str(boundary / "2022_06_05_fwi_comp.nc"))

    def test_real_daily_fwi_produces_readable_trigger_file(self):
        variables = (
            ("2t", 293.15, "K"),
            ("2d", 283.15, "K"),
            ("10u", 1.0, "m s-1"),
            ("10v", 1.0, "m s-1"),
            ("avg_tprate", 0.00001, "kg m-2 s-1"),
        )
        for day in ("2022-06-05", "2022-06-06"):
            times = np.arange(
                np.datetime64(day, "h"), np.datetime64(day, "h") + np.timedelta64(24, "h"),
                np.timedelta64(1, "h"),
            )
            for name, value, units in variables:
                data = np.full((24, len(self.lat), len(self.lon)), value, dtype="float32")
                dataset = xr.Dataset(
                    {name: (("time", "lat", "lon"), data, {"units": units})},
                    coords={"time": times, "lat": self.lat, "lon": self.lon},
                    attrs={"experiment": "hist"},
                )
                dataset.to_netcdf(self.input_dir / f"{day.replace('-', '_')}_{name}.nc")
        args = self.arguments()
        args.initialize = True
        args.overwintering = False
        args.no_annual = True
        with patch.object(run_wildfires, "run_wise"), contextlib.redirect_stdout(io.StringIO()):
            run_wildfires.run_wildfires(args)
        self.assertTrue(self.path.is_file())
        self.assertTrue((self.fwi_dir / "2022_06_06_fwi_comp.nc").is_file())
        self.assertTrue((self.fwi_dir / "FWI_Const1.nc").is_file())


if __name__ == "__main__":
    unittest.main()
