import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from test_netcdf_time import wise


class FWIInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wise-fwi-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fwi_file = self.root / "1990_07_01_fwi_comp.nc"
        self.dataset = xr.Dataset(
            {
                "FFMC": (("time", "lat", "lon"), [[[80, 81, 82], [83, 84, 85]]]),
                "DMC": (("time", "lat", "lon"), [[[10, 11, 12], [13, 14, 15]]]),
                "DC": (("time", "lat", "lon"), [[[100, 101, 102], [103, 104, 105]]]),
            },
            coords={
                "time": [np.datetime64("1990-07-01T12:00")],
                "lat": [40.025, 64.025],
                "lon": [0, 24.15, 351.9],
            },
            attrs={"model": "DO_NOT_PROPAGATE"},
        ).astype("float32")

    def read_codes(self, coords=((40.041393, -8.114061),), diagnostics=None):
        self.dataset.to_netcdf(self.fwi_file)
        with contextlib.redirect_stdout(io.StringIO()):
            return wise.read_fwi_starting_codes(
                str(self.fwi_file), "1990-07-02", coords, diagnostics=diagnostics
            )

    def test_finland_and_portugal_use_different_cells(self):
        self.assertEqual(
            self.read_codes(((64.007044, 24.152986), (40.041393, -8.114061))),
            [
                {"ffmc": 84.0, "dmc": 14.0, "dc": 104.0},
                {"ffmc": 82.0, "dmc": 12.0, "dc": 102.0},
            ],
        )

    def test_negative_longitudes_and_descending_latitude(self):
        self.dataset = self.dataset.assign_coords(lon=[0, 24.15, -8.1]).isel(
            lat=slice(None, None, -1)
        )
        self.assertEqual(self.read_codes()[0]["ffmc"], 82.0)

    def test_circular_selection_near_greenwich(self):
        self.assertEqual(self.read_codes(((40.025, -0.01),))[0]["ffmc"], 80.0)

    def test_uses_matching_local_date_not_nearest_timestamp(self):
        other = self.dataset.assign_coords(time=[np.datetime64("1990-06-30T12:00")])
        other = other.assign(FFMC=other.FFMC + 10)
        self.dataset = xr.concat([other, self.dataset], dim="time")
        self.assertEqual(self.read_codes()[0]["ffmc"], 82.0)

    def test_previous_day_handles_calendar_boundaries(self):
        for simulation_date, fwi_date in (
            ("1990-07-01", "1990-06-30"),
            ("1991-01-01", "1990-12-31"),
            ("1992-03-01", "1992-02-29"),
            ("1992-02-29", "1992-02-28"),
            ("1991-03-01", "1991-02-28"),
        ):
            with self.subTest(simulation_date=simulation_date):
                self.dataset.assign_coords(
                    time=[np.datetime64(fwi_date + "T12:00")]
                ).to_netcdf(self.fwi_file)
                diagnostics = {}
                with contextlib.redirect_stdout(io.StringIO()):
                    codes = wise.read_fwi_starting_codes(
                        str(self.fwi_file), simulation_date, [(40.041393, -8.114061)],
                        diagnostics=diagnostics,
                    )
                self.assertEqual(codes[0]["ffmc"], 82.0)
                self.assertEqual(diagnostics["required_fwi_date"], fwi_date)
                self.assertEqual(diagnostics["simulation_date"], simulation_date)
                self.assertEqual(diagnostics["date_rule"], "previous_day")

    def test_directory_selects_previous_day_component_file(self):
        self.dataset.to_netcdf(self.fwi_file)
        self.dataset.assign(FFMC=self.dataset.FFMC + 10).assign_coords(
            time=[np.datetime64("1990-07-02T12:00")]
        ).to_netcdf(self.root / "1990_07_02_fwi_comp.nc")
        (self.root / "1990_07_01_fwi.nc").write_text("not the component file")
        selected = wise.resolve_fwi_file("1990-07-02", fwi_dir=str(self.root))
        self.assertEqual(selected, str(self.fwi_file))
        diagnostics = {}
        with contextlib.redirect_stdout(io.StringIO()):
            codes = wise.read_fwi_starting_codes(
                selected, "1990-07-02", [(40.041393, -8.114061)],
                diagnostics=diagnostics,
            )
        self.assertEqual(codes[0]["ffmc"], 82.0)
        self.assertEqual(diagnostics["selected_file"], str(self.fwi_file))
        for run_date, previous_name in (
            ("1991-01-01", "1990_12_31_fwi_comp.nc"),
            ("1992-03-01", "1992_02_29_fwi_comp.nc"),
            ("2016-08-21", "2016_08_20_fwi_comp.nc"),
        ):
            self.assertEqual(
                wise.resolve_fwi_file(run_date, fwi_dir=self.root),
                str(self.root / previous_name),
            )

    def test_missing_directory_file_preserves_warning_policy(self):
        for directory in (self.root, self.root / "missing"):
            selected = wise.resolve_fwi_file("1990-07-02", fwi_dir=directory)
            diagnostics = {}
            with contextlib.redirect_stdout(io.StringIO()):
                codes = wise.read_fwi_starting_codes(
                    selected, "1990-07-02", [(40, -8)], diagnostics=diagnostics,
                )
            self.assertIsNone(codes)
            self.assertEqual(diagnostics["status"], "template_fallback")
            self.assertIn("1990_07_01_fwi_comp.nc", diagnostics["warnings"][0])

    def test_conflicting_or_invalid_fwi_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            wise.resolve_fwi_file("1990-07-02", fwi_file="file.nc", fwi_dir=self.root)
        self.fwi_file.write_text("not a directory")
        with self.assertRaisesRegex(ValueError, "not a directory"):
            wise.resolve_fwi_file("1990-07-02", fwi_dir=self.fwi_file)
        self.assertEqual(
            wise.resolve_fwi_file("1990-07-02", fwi_file=self.fwi_file), self.fwi_file,
        )
        self.assertIsNone(wise.resolve_fwi_file("1990-07-02"))

    def test_same_day_file_is_rejected(self):
        self.dataset.to_netcdf(self.fwi_file)
        with self.assertRaisesRegex(ValueError, "1990-06-30.*previous day"):
            wise.read_fwi_starting_codes(
                str(self.fwi_file), "1990-07-01", [(40, -8)]
            )

    def test_rejects_missing_or_duplicate_date(self):
        for times in (
            ["1990-06-30T12:00"],
            ["1990-07-01T00:00", "1990-07-01T12:00"],
        ):
            with self.subTest(times=times):
                self.dataset = self.dataset.isel(time=[0] * len(times)).assign_coords(
                    time=np.array(times, dtype="datetime64[ns]")
                )
                with self.assertRaisesRegex(ValueError, "exactly one FWI time"):
                    self.read_codes()

    def test_invalid_codes_warn_and_skip_only_affected_cells(self):
        for name, value in (
            ("FFMC", np.nan), ("FFMC", np.inf), ("FFMC", 101.1),
            ("FFMC", -1), ("DMC", -1), ("DC", -1),
        ):
            with self.subTest(name=name, value=value):
                original = self.dataset.copy(deep=True)
                self.dataset[name].values[0, 0, 2] = value
                diagnostics = {}
                codes = self.read_codes(
                    ((40.041393, -8.114061), (64.007044, 24.152986)),
                    diagnostics=diagnostics,
                )
                self.assertIsNone(codes[0])
                self.assertEqual(codes[1]["ffmc"], 84.0)
                self.assertEqual(diagnostics["status"], "partial")
                warning = diagnostics["warnings"][0]
                for text in (
                    f"{name}=", "area1", "1990-07-01", "ignition=", "cell=",
                    str(self.fwi_file), "Skipping this area",
                ):
                    self.assertIn(text, warning)
                self.dataset = original

    def test_rejects_missing_variable_and_bad_coordinates(self):
        original = self.dataset
        self.dataset = self.dataset.drop_vars("DC")
        with self.assertRaisesRegex(ValueError, "expected DC"):
            self.read_codes()
        self.dataset = original.assign_coords(lat=[np.nan, 64.025])
        with self.assertRaisesRegex(ValueError, "invalid lat"):
            self.read_codes()

    def test_missing_file_is_not_silently_ignored(self):
        diagnostics = {}
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            codes = wise.read_fwi_starting_codes(
                str(self.fwi_file), "1990-07-02", [(40, -8)],
                diagnostics=diagnostics,
            )
        self.assertIsNone(codes)
        self.assertEqual(diagnostics["status"], "template_fallback")
        self.assertIn("WARNING: FWI file not found", log.getvalue())
        self.assertIn(str(self.fwi_file), diagnostics["warnings"][0])
        self.assertIn("template FFMC", diagnostics["warnings"][0])

    def test_corrupt_file_does_not_fall_back(self):
        self.fwi_file.write_text("not a NetCDF")
        with self.assertRaises((ValueError, OSError)):
            wise.read_fwi_starting_codes(
                str(self.fwi_file), "1990-07-01", [(40, -8)]
            )

    def test_cli_forwards_optional_file(self):
        arguments = [
            "run_wise.py", "-year_start", "1990", "-month_start", "07",
            "-day_start", "01", "-year_end", "1990", "-month_end", "07",
            "-day_end", "01",
        ]
        for supplied in (None, str(self.fwi_file)):
            cli = arguments + (["--fwi-file", supplied] if supplied else [])
            with self.subTest(fwi_file=supplied), \
                    patch.object(sys, "argv", cli), \
                    patch.object(wise, "run_wise") as run:
                wise.main()
            self.assertEqual(run.call_args.kwargs["fwi_file"], supplied)
            self.assertIsNone(run.call_args.kwargs["fwi_dir"])
        with patch.object(sys, "argv", arguments + ["--fwi-dir", str(self.root)]), \
                patch.object(wise, "run_wise") as run:
            wise.main()
        self.assertEqual(run.call_args.kwargs["fwi_dir"], str(self.root))
        self.assertIsNone(run.call_args.kwargs["fwi_file"])
        with patch.object(sys, "argv", arguments + [
            "--fwi-dir", str(self.root), "--fwi-file", str(self.fwi_file),
        ]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                wise.main()
        self.assertEqual(error.exception.code, 2)

    def template(self):
        time = {"time": "old", "timezone": "unchanged"}
        return {
            "project": {
                "scenarios": {"scenarioData": [{
                    "startTime": copy.deepcopy(time),
                    "endTime": copy.deepcopy(time),
                    "temporalConditions": {"daily": [{
                        name: copy.deepcopy(time)
                        for name in ("localStartTime", "startTime", "endTime")
                    }]},
                }]},
                "ignitions": {"ignitionData": [{
                    "startTime": copy.deepcopy(time),
                    "ignitions": {"ignitions": [{
                        "polygon": {"polygon": {"points": [{
                            "x": {"value": 0}, "y": {"value": 0},
                        }]}}
                    }]},
                }]},
                "outputs": {
                    "grids": [
                        {"exportTime": copy.deepcopy(time)} for _ in range(7)
                    ],
                    "vectors": [{"perimeterTime": {
                        "startTime": copy.deepcopy(time),
                        "endTime": copy.deepcopy(time),
                    }}],
                    "summaries": [{}],
                },
                "stations": {"wxStationData": [{"streams": [{"condition": {
                    "startTime": copy.deepcopy(time),
                    "startingCodes": {
                        "ffmc": {"value": 89.0},
                        "dmc": {"value": 58.0},
                        "dc": {"value": 482.0},
                        "precipitation": {"value": 0.0},
                    },
                    "hffmc": {"value": 94.0},
                    "hffmcTime": {"time": "17:00:00:00"},
                    "hffmcMethod": "LAWSON",
                }}]}]},
                "grid": {"fuelMap": {}, "elevation": {}, "projection": {}},
            }
        }

    def test_all_nine_generated_jobs_and_unchanged_hourly_settings(self):
        template = self.template()
        template_path = self.root / "template.fgmj"
        template_path.write_text(json.dumps(template))
        original_open = open

        def redirected_open(path, *args, **kwargs):
            path = str(path)
            if path.startswith("/testjobs/"):
                path = template_path
            elif path.startswith("/wise_output/"):
                path = self.root / path.removeprefix("/wise_output/")
            return original_open(path, *args, **kwargs)

        coords = [(64.007044, 24.152986)] * 3 + [(40.041393, -8.114061)] * 6
        values = self.read_codes(coords)
        for area in range(1, 10):
            (self.root / f"area{area}").mkdir()
        for supplied in (None, values, [None] + values[1:]):
            first_job = self.root / "area1/job.fgmj"
            if first_job.exists():
                first_job.unlink()
            with self.subTest(fwi_enabled=supplied is not None), \
                    patch("builtins.open", side_effect=redirected_open), \
                    contextlib.redirect_stdout(io.StringIO()):
                wise.modify_fgmj(
                    "1990-07-01T01:00:00", "1990-07-01T20:00:00",
                    "1990-07-01T10:00:00", "1990-07-01T21:00:00",
                    *(coordinate for pair in coords for coordinate in pair),
                    fwi_starting_codes=supplied,
                )
            for area in range(1, 10):
                if supplied is not None and supplied[area - 1] is None:
                    self.assertFalse((self.root / f"area{area}/job.fgmj").exists())
                    continue
                job = json.loads((self.root / f"area{area}/job.fgmj").read_text())
                condition = job["project"]["stations"]["wxStationData"][0]["streams"][0]["condition"]
                original = template["project"]["stations"]["wxStationData"][0]["streams"][0]["condition"]
                expected = copy.deepcopy(original["startingCodes"])
                if supplied is not None:
                    for name in ("ffmc", "dmc", "dc"):
                        expected[name]["value"] = values[area - 1][name]
                self.assertEqual(condition["startingCodes"], expected)
                for name in ("hffmc", "hffmcTime", "hffmcMethod"):
                    self.assertEqual(condition[name], original[name])

    def test_weather_preparation_passes_codes_to_job_creation(self):
        self.dataset.assign_coords(
            time=[np.datetime64("1990-06-30T12:00")]
        ).to_netcdf(self.fwi_file)
        climate_path = self.root / "climate.nc"
        xr.Dataset(
            {
                name: (("time", "lat", "lon"), np.full((24, 2, 2), value))
                for name, value in (
                    ("2t", 290.0), ("2d", 280.0), ("10u", 1.0),
                    ("10v", 1.0), ("tp", 0.0),
                )
            },
            coords={
                "time": np.arange(
                    np.datetime64("1990-07-01T00"),
                    np.datetime64("1990-07-02T00"),
                    np.timedelta64(1, "h"),
                ),
                "lat": [40.025, 64.025], "lon": [-8.1, 24.15],
            },
        ).to_netcdf(climate_path)
        for supplied in (None, str(self.fwi_file)):
            with self.subTest(fwi_file=supplied), \
                    patch.object(wise, "modify_fgmj", return_value=tuple(range(9))) as modify, \
                    patch.object(wise.os, "makedirs"), \
                    patch.object(wise.pd.DataFrame, "to_csv"), \
                    contextlib.redirect_stdout(io.StringIO()):
                wise.ncdf_edits_multiarea(str(climate_path), fwi_file=supplied)
            codes = modify.call_args.kwargs["fwi_starting_codes"]
            if supplied is None:
                self.assertIsNone(codes)
            else:
                self.assertEqual(len(codes), 9)
                self.assertEqual(codes[0]["ffmc"], 84.0)
                self.assertEqual(codes[3]["ffmc"], 82.0)
        self.dataset.to_netcdf(self.fwi_file)
        with patch.object(wise, "modify_fgmj") as modify, \
                patch.object(wise.pd.DataFrame, "to_csv"), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "1990-06-30"):
                wise.ncdf_edits_multiarea(
                    str(climate_path), fwi_file=str(self.fwi_file),
                    simulation_date="1990-07-01",
                )
            modify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
