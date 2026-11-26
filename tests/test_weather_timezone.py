import contextlib
import csv
from datetime import timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import test_fwi_initialization as fwi_tests
from test_netcdf_time import wise


CASES = (
    ("1990-02-01", ("EET", 65544, 2), ("UTC", 65553, 0)),
    ("1990-07-06", ("EEST", 131081, 3), ("WEST", 131089, 1)),
    ("1990-03-25", ("EEST", 131081, 3), ("WEST", 131089, 1)),
    ("1990-09-30", ("EET", 65544, 2), ("UTC", 65553, 0)),
    ("2026-03-29", ("EEST", 131081, 3), ("WEST", 131089, 1)),
    ("2026-10-25", ("EET", 65544, 2), ("UTC", 65553, 0)),
    ("1993-02-01", ("EET", 65544, 2), ("CET", 65541, 1)),
    ("1993-07-06", ("EEST", 131081, 3), ("CEST", 131079, 2)),
    ("1992-09-27", ("EET", 65544, 2), ("CET", 65541, 1)),
    ("1996-03-31", ("EEST", 131081, 3), ("WEST", 131089, 1)),
    ("1990-12-31", ("EET", 65544, 2), ("UTC", 65553, 0)),
    ("2049-07-06", ("EEST", 131081, 3), ("WEST", 131089, 1)),
)


def climate_dataset(day, hours=24):
    times = pd.date_range(day, periods=hours, freq="h")
    return xr.Dataset(
        {
            name: (("time", "lat", "lon"), values[:, None, None])
            for name, values in (
                ("2t", 273.15 + np.arange(hours, dtype=float)),
                ("2d", np.full(hours, 270.0)),
                ("10u", np.full(hours, 1.0)),
                ("10v", np.full(hours, 1.0)),
                ("tp", np.arange(hours, dtype=float) / 10),
            )
        },
        coords={"time": times, "lat": [50.0], "lon": [0.0]},
    )


class WeatherTimezoneTests(unittest.TestCase):
    def test_timezone_selection(self):
        for day, finland, portugal in CASES:
            for area, expected in ((0, finland), (3, portugal)):
                with self.subTest(day=day, area=area):
                    code, zone_id, zone, _ = wise.run_timezone(day, area)
                    self.assertEqual((code, zone_id), expected[:2])
                    self.assertEqual(zone.utcoffset(None), timedelta(hours=expected[2]))

    def test_weather_and_all_nine_jobs(self):
        for day, finland, portugal in CASES:
            with self.subTest(day=day), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "climate.nc"
                climate_dataset(day).to_netcdf(source)
                template = root / "template.fgmj"
                template.write_text(json.dumps(fwi_tests.FWIInitializationTests().template()))
                original_open = open
                original_to_csv = pd.DataFrame.to_csv

                def redirected_open(path, *args, **kwargs):
                    path = str(path)
                    if path.startswith("/testjobs/"):
                        path = template
                    elif path.startswith("/wise_output/"):
                        path = root / path.removeprefix("/wise_output/")
                        path.parent.mkdir(parents=True, exist_ok=True)
                    return original_open(path, *args, **kwargs)

                def write_weather(frame, path, *args, **kwargs):
                    destination = root / str(path).removeprefix("/wise_output/")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    return original_to_csv(frame, destination, *args, **kwargs)

                log = io.StringIO()
                with patch("builtins.open", side_effect=redirected_open), \
                        patch.object(wise.os.path, "exists", return_value=True), \
                        patch.object(pd.DataFrame, "to_csv", new=write_weather), \
                        contextlib.redirect_stdout(log):
                    wise.ncdf_edits_multiarea(str(source), simulation_date=day)
                if day in ("1990-03-25", "1990-09-30", "2026-03-29", "2026-10-25"):
                    self.assertIn("WARNING:", log.getvalue())
                for area in range(1, 10):
                    expected = finland if area <= 3 else portugal
                    with (root / f"area{area}/Inputs/weather.txt").open() as stream:
                        weather = list(csv.DictReader(stream))
                    self.assertEqual(len(weather), 24)
                    actual_times = pd.DatetimeIndex([
                        pd.to_datetime(row["HOURLY"], format="%d/%m/%Y")
                        for row in weather
                    ]) + pd.to_timedelta([int(row["HOUR"]) for row in weather], unit="h")
                    expected_times = pd.date_range(day, periods=24, freq="h")
                    expected_times += pd.Timedelta(hours=expected[2])
                    self.assertTrue(actual_times.equals(expected_times))
                    self.assertEqual([float(row["TEMP"]) for row in weather], list(range(24)))
                    np.testing.assert_allclose(
                        [float(row["PRECIP"]) for row in weather], np.arange(24) / 10,
                    )
                    job = json.loads((root / f"area{area}/job.fgmj").read_text())
                    project = job["project"]
                    self.assertEqual(project["timeZoneSettings"]["timezoneIndex"], expected[1])

                    def check_zones(value):
                        if isinstance(value, dict):
                            if "timezone" in value or "timezoneId" in value:
                                self.assertEqual(value["timezone"], expected[0])
                                self.assertEqual(value["timezoneId"], expected[1])
                            for child in value.values():
                                check_zones(child)
                        elif isinstance(value, list):
                            for child in value:
                                check_zones(child)

                    check_zones(project)
                    scenario = project["scenarios"]["scenarioData"][0]
                    ignition = project["ignitions"]["ignitionData"][0]["startTime"]["time"]
                    self.assertEqual(scenario["startTime"]["time"], ignition)
                    self.assertEqual(pd.Timestamp(ignition).hour, 10)
                    end = scenario["endTime"]["time"]
                    self.assertEqual(pd.Timestamp(end).hour, 21)
                    self.assertEqual(pd.Timestamp(ignition).utcoffset(), timedelta(hours=expected[2]))
                    for grid in project["outputs"]["grids"]:
                        self.assertEqual(grid["exportTime"]["time"], end)
                    perimeter = project["outputs"]["vectors"][0]["perimeterTime"]
                    self.assertEqual(perimeter["startTime"]["time"], ignition)
                    self.assertEqual(perimeter["endTime"]["time"], end)
                    condition = project["stations"]["wxStationData"][0]["streams"][0]["condition"]
                    self.assertEqual(
                        pd.Timestamp(condition["startTime"]["time"]).tz_convert("UTC"),
                        pd.Timestamp(day, tz="UTC"),
                    )

    def test_multiday_transition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "climate.nc"
            climate_dataset("2026-03-28", hours=48).to_netcdf(path)
            with self.assertRaisesRegex(ValueError, "split the run into separate dates"):
                wise.ncdf_edits_multiarea(str(path))

    def test_combined_run_uses_requested_local_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "climate.nc"
            climate_dataset("2022-06-06").to_netcdf(path)
            template = root / "template.fgmj"
            template.write_text(json.dumps(fwi_tests.FWIInitializationTests().template()))
            original_open = open

            def redirected_open(path, *args, **kwargs):
                path = str(path)
                if path.startswith("/testjobs/"):
                    path = template
                elif path.startswith("/wise_output/"):
                    path = root / path.removeprefix("/wise_output/")
                    path.parent.mkdir(parents=True, exist_ok=True)
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=redirected_open), \
                    patch.object(pd.DataFrame, "to_csv"), \
                    patch.object(wise.os.path, "exists", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                wise.ncdf_edits_multiarea(
                    str(path), simulation_date="2022-06-06",
                    scenario_start_hour=8, scenario_end_hour=20,
                )
            for area in range(1, 10):
                project = json.loads((root / f"area{area}/job.fgmj").read_text())["project"]
                scenario = project["scenarios"]["scenarioData"][0]
                start = pd.Timestamp(scenario["startTime"]["time"])
                end = pd.Timestamp(scenario["endTime"]["time"])
                self.assertEqual((start.hour, end.hour), (8, 20))
                self.assertEqual(start.date(), end.date())
                self.assertEqual(start.date().isoformat(), "2022-06-06")
                self.assertEqual(start.utcoffset(), timedelta(hours=3 if area <= 3 else 1))

    def test_nonconsecutive_weather_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "climate.nc"
            climate_dataset("1990-07-06").isel(time=[0, 2, 3]).to_netcdf(path)
            with self.assertRaisesRegex(ValueError, "consecutive, unique whole-hour"):
                wise.ncdf_edits_multiarea(str(path))

    def test_multiday_run_keeps_local_end_on_final_input_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "climate.nc"
            climate_dataset("2032-02-29", hours=48).to_netcdf(path)
            with patch.object(wise, "modify_fgmj", return_value=tuple(range(9))) as modify, \
                    patch.object(pd.DataFrame, "to_csv") as write, \
                    patch.object(wise.os.path, "exists", return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                wise.ncdf_edits_multiarea(str(path))
            self.assertEqual(modify.call_args.args[2], "2032-02-29T10:00:00")
            self.assertEqual(modify.call_args.args[3], "2032-03-01T21:00:00")
            self.assertEqual(len(modify.call_args.kwargs["weather_starts"]), 9)
            self.assertEqual(write.call_count, 9)

    def test_missing_timezone_data_gives_actionable_error(self):
        with patch.object(wise, "ZoneInfo", side_effect=wise.ZoneInfoNotFoundError("missing")):
            with self.assertRaisesRegex(RuntimeError, "install.*tzdata.*bind"):
                wise.run_timezone("1990-07-06", 0)


if __name__ == "__main__":
    unittest.main()
