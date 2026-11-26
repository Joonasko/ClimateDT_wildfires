# loading packages

import argparse
import json
import numpy as np
import os
import pandas as pd
import subprocess
import xarray as xr
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import geopandas as gp
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.transform import xy

WISE_TIMEOUT = 10800  # Per-area limit in seconds: 1 hour.

AREA_COORDS = (
    (64.007044, 24.152986),
    (63.050609, 29.889436),
    (63.436092, 30.541259),
    (40.041393, -8.114061),
    (40.041393, -8.114061),
    (40.196173, -8.137504),
    (40.196173, -8.137504),
    (40.125040, -8.113300),
    (40.125040, -8.113300),
)


def run_timezone(simulation_date, area_index, ignition_hour=10):
    zone_name = "Europe/Helsinki" if area_index < 3 else "Europe/Lisbon"
    ignition = (pd.Timestamp(simulation_date).normalize() + pd.Timedelta(hours=ignition_hour))
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as error:
        raise RuntimeError(
            "Timezone database unavailable: install the project's tzdata dependency "
            "or bind /usr/share/zoneinfo:/usr/share/zoneinfo:ro into the container"
        ) from error
    ignition = ignition.tz_localize(zone)
    offset = ignition.utcoffset()
    standard_offset = offset - ignition.dst()
    if standard_offset == timedelta(hours=2):
        code, zone_id = ("EEST", 131081) if ignition.dst() else ("EET", 65544)
    elif standard_offset == timedelta(hours=1):
        code, zone_id = ("CEST", 131079) if ignition.dst() else ("CET", 65541)
    elif standard_offset == timedelta(0):
        # WET 65682 is advertised by Builder but unsupported by this runtime.
        code, zone_id = ("WEST", 131089) if ignition.dst() else ("UTC", 65553)
    else:
        raise ValueError(f"Unsupported WISE timezone for {zone_name}: {ignition}")
    return code, zone_id, timezone(offset), zone_name


def set_job_timezone(value, code, zone_id):
    if isinstance(value, dict):
        if "timezone" in value or "timezoneId" in value:
            value.update(timezone=code, timezoneId=zone_id)
        if "timezoneIndex" in value:
            value["timezoneIndex"] = zone_id
        for child in value.values():
            set_job_timezone(child, code, zone_id)
    elif isinstance(value, list):
        for child in value:
            set_job_timezone(child, code, zone_id)


def resolve_fwi_file(simulation_date, fwi_file=None, fwi_dir=None):
    if fwi_file is not None and fwi_dir is not None:
        raise ValueError("Supply either fwi_file or fwi_dir, not both")
    if fwi_dir is None:
        return fwi_file
    if os.path.exists(fwi_dir) and not os.path.isdir(fwi_dir):
        raise ValueError(f"FWI directory path is not a directory: {fwi_dir}")
    previous_date = pd.Timestamp(simulation_date) - pd.Timedelta(days=1)
    return os.path.join(fwi_dir, previous_date.strftime("%Y_%m_%d_fwi_comp.nc"))


def read_fwi_starting_codes(fwi_file, simulation_date, area_coords, diagnostics=None):
    if diagnostics is None:
        diagnostics = {}
    diagnostics.update(status="reading", warnings=[], areas={})
    simulation_date_label = pd.Timestamp(simulation_date).strftime("%Y-%m-%d")
    date_label = (pd.Timestamp(simulation_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    diagnostics.update(
        simulation_date=simulation_date_label,
        required_fwi_date=date_label,
        date_rule="previous_day",
        selected_file=os.path.abspath(fwi_file),
    )
    try:
        dataset = xr.open_dataset(fwi_file)
    except FileNotFoundError:
        message = (
            f"FWI file not found: {fwi_file}; required previous-day date={date_label}, "
            f"simulation date={simulation_date_label}. "
            "Continuing all areas with template FFMC, DMC and DC starting values."
        )
        print(f"WARNING: {message}")
        diagnostics.update(status="template_fallback", warnings=[message])
        return None
    with dataset:
        for name in ("lat", "lon", "time"):
            if name not in dataset.coords or dataset[name].dims != (name,):
                raise ValueError(f"{fwi_file}: expected a one-dimensional {name} coordinate")
        for name in ("FFMC", "DMC", "DC"):
            if name not in dataset or set(dataset[name].dims) != {"time", "lat", "lon"}:
                raise ValueError(f"{fwi_file}: expected {name}(time, lat, lon)")

        matching_dates = np.flatnonzero(
            dataset.time.dt.strftime("%Y-%m-%d").values == date_label
        )
        if len(matching_dates) != 1:
            raise ValueError(
                f"{fwi_file}: expected exactly one FWI time for {date_label} "
                f"(previous day of simulation {simulation_date_label}), "
                f"found {len(matching_dates)}"
            )
        latitudes = dataset.lat.values
        longitudes = dataset.lon.values
        for name, values, lower, upper in (
            ("lat", latitudes, -90, 90),
            ("lon", longitudes, -180, 360),
        ):
            if (
                values.size == 0
                or not np.isfinite(values).all()
                or np.any(values < lower)
                or np.any(values > upper)
            ):
                raise ValueError(f"{fwi_file}: invalid {name} coordinates")

        starting_codes = []
        for area, (latitude, longitude) in enumerate(area_coords, start=1):
            # Circular distance handles both longitude conventions and the seam.
            lon_distance = np.abs((longitudes - longitude + 180) % 360 - 180)
            cell = dataset.isel(
                time=int(matching_dates[0]),
                lat=int(np.argmin(np.abs(latitudes - latitude))),
                lon=int(np.argmin(lon_distance)),
            )
            values = {}
            invalid = []
            for name in ("FFMC", "DMC", "DC"):
                value = float(cell[name].item())
                if not np.isfinite(value) or value < 0 or (name == "FFMC" and value > 101):
                    invalid.append(f"{name}={value}")
                values[name.lower()] = value
            if invalid:
                message = (
                    f"{fwi_file}: missing or invalid {', '.join(invalid)} for area{area}, "
                    f"date={date_label}, ignition=({latitude}, {longitude}), "
                    f"cell=({float(cell.lat)}, {float(cell.lon)}). Skipping this area."
                )
                print(f"WARNING: {message}")
                diagnostics["warnings"].append(message)
                diagnostics["areas"][area] = {"status": "invalid_values", "error": message}
                starting_codes.append(None)
                continue
            diagnostics["areas"][area] = {
                "status": "valid",
                "latitude": float(cell.lat),
                "longitude": float(cell.lon),
                "starting_codes": values,
            }
            print(
                f"FWI area{area}: file={fwi_file}, date={date_label}, "
                f"ignition=({latitude}, {longitude}), "
                f"cell=({float(cell.lat)}, {float(cell.lon)}), codes={values}"
            )
            starting_codes.append(values)
    diagnostics["status"] = "partial" if diagnostics["warnings"] else "valid"
    return starting_codes


def modify_fgmj(
    scenario_start,
    scenario_end,
    ignition_start,
    ignition_end,
    ignition_y_1,
    ignition_x_1,
    ignition_y_2,
    ignition_x_2,
    ignition_y_3,
    ignition_x_3,
    ignition_y_4,
    ignition_x_4,
    ignition_y_5,
    ignition_x_5,
    ignition_y_6,
    ignition_x_6,
    ignition_y_7,
    ignition_x_7,
    ignition_y_8,
    ignition_x_8,
    ignition_y_9,
    ignition_x_9,
    fwi_starting_codes=None,
    weather_starts=None,
):
    
    print("running .fgmj modifier")
    if fwi_starting_codes is not None and len(fwi_starting_codes) != 9:
        raise ValueError("Expected FWI starting codes for all nine areas")
    if weather_starts is not None and len(weather_starts) != 9:
        raise ValueError("Expected weather start times for all nine areas")
    weather_start_utc = pd.Timestamp(scenario_start).tz_localize("UTC")
    # set scenario names
    scen_name_1 = "scen_fin_kalajoki"
    scen_name_2 = "scen_fin_koli"
    scen_name_3 = "scen_fin_lieksa"
    scen_name_4 = "scen_por_1_norm"
    scen_name_5 = "scen_por_1_break"
    scen_name_6 = "scen_por_2_norm"
    scen_name_7 = "scen_por_2_break"
    scen_name_8 = "scen_por_3_norm"
    scen_name_9 = "scen_por_3_break"

    # set input fgmj path and read the fgmj files
    fgmj_fin_path = "/testjobs/testjobs/job_fin.fgmj"
    fgmj_por_path = "/testjobs/testjobs/job_por.fgmj"

    # Load fgmj files (3 Finland, 6 Portugal)
    fgmj_data_list = []
    for i in range(3):
        with open(fgmj_fin_path, "r") as f:
            fgmj_data_list.append(json.load(f))
    for i in range(6):
        with open(fgmj_por_path, "r") as f:
            fgmj_data_list.append(json.load(f))





    # set variables
    scenario_start = ignition_start
    scenario_end = ignition_end
    local_start_time = ignition_start
    start_time = ignition_start
    end_time = ignition_end
    ignition_start = ignition_start
    output_time = scenario_end

    # function for replacing values in dictionary
    def replace_in_dict(data, find, replace):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    replace_in_dict(value, find, replace)
                elif isinstance(value, str):
                    data[key] = value.replace(find, replace)

        elif isinstance(data, list):
            for index, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    replace_in_dict(value, find, replace)
                elif isinstance(value, str):
                    data[index] = value.replace(find, replace)

    # function for editing the job.fgmj files
    def create_job(
        data_in, job_name, scen_name, ign_lon, ign_lat, weather_path, spat_path,
        starting_codes, area_index,
    ):
        code, zone_id, fixed_zone, _ = run_timezone(
            ignition_start[:10], area_index, pd.Timestamp(ignition_start).hour
        )
        set_job_timezone(data_in, code, zone_id)
        data_in["project"]["timeZoneSettings"] = {
            "version": 1, "timezoneIndex": zone_id,
        }

        def local_time(value):
            return pd.Timestamp(value).tz_localize(fixed_zone).isoformat()

        weather_start = (
            weather_starts[area_index] if weather_starts is not None
            else weather_start_utc.tz_convert(fixed_zone).isoformat()
        )
        data_in["project"]["projectStartTime"] = {
            "time": weather_start, "timezone": code, "timezoneId": zone_id,
        }
        data_in["project"]["scenarios"]["scenarioData"][0]["startTime"]["time"] = (
            local_time(scenario_start)
        )

        data_in["project"]["scenarios"]["scenarioData"][0]["endTime"]["time"] = (
            local_time(scenario_end)
        )

        data_in["project"]["scenarios"]["scenarioData"][0]["temporalConditions"][
            "daily"
        ][0]["localStartTime"]["time"] = local_time(local_start_time)

        data_in["project"]["scenarios"]["scenarioData"][0]["temporalConditions"][
            "daily"
        ][0]["startTime"]["time"] = local_time(start_time)

        data_in["project"]["scenarios"]["scenarioData"][0]["temporalConditions"][
            "daily"
        ][0]["endTime"]["time"] = local_time(end_time)

        data_in["project"]["ignitions"]["ignitionData"][0]["startTime"]["time"] = (
            local_time(ignition_start)
        )

        data_in["project"]["ignitions"]["ignitionData"][0]["ignitions"]["ignitions"][0][
            "polygon"
        ]["polygon"]["points"][0]["x"]["value"] = ign_lon

        data_in["project"]["ignitions"]["ignitionData"][0]["ignitions"]["ignitions"][0][
            "polygon"
        ]["polygon"]["points"][0]["y"]["value"] = ign_lat

        data_in["project"]["outputs"]["grids"][0]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][1]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][2]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][3]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][4]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][5]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["grids"][6]["exportTime"]["time"] = local_time(output_time)

        data_in["project"]["outputs"]["vectors"][0]["perimeterTime"]["startTime"][
            "time"
        ] = local_time(ignition_start)

        data_in["project"]["outputs"]["vectors"][0]["perimeterTime"]["endTime"][
            "time"
        ] = local_time(output_time)

        data_in["project"]["stations"]["wxStationData"][0]["streams"][0]["condition"][
            "startTime"
        ]["time"] = weather_start
        if starting_codes is not None:
            condition = data_in["project"]["stations"]["wxStationData"][0]["streams"][0]["condition"]
            for name in ("ffmc", "dmc", "dc"):
                condition["startingCodes"][name]["value"] = starting_codes[name]
        
        
        data_in["project"]["outputs"]["summaries"][0]["filename"] = (
            "/wise_output/scen0/summary.txt"
        )
        
        data_in["project"]["outputs"]["grids"][0]["filename"] = (
            "/wise_output/scen0/max_intensity.tif"
        )
        data_in["project"]["outputs"]["grids"][1]["filename"] = (
            "/wise_output/scen0/max_flame_length.tif"
        )
        data_in["project"]["outputs"]["grids"][2]["filename"] = (
            "/wise_output/scen0/burn_grid.tif"
        )
        data_in["project"]["outputs"]["grids"][3]["filename"] = (
            "/wise_output/scen0/crown_fuel_consumed.tif"
        )
        data_in["project"]["outputs"]["grids"][4]["filename"] = (
            "/wise_output/scen0/surface_fuel_consumed.tif"
        )
        data_in["project"]["outputs"]["grids"][5]["filename"] = (
            "/wise_output/scen0/max_crown_fraction_burned.tif"
        )
        data_in["project"]["outputs"]["grids"][6]["filename"] = (
            "/wise_output/scen0/total_fuel_consumption.tif"
        )
        data_in["project"]["outputs"]["vectors"][0]["filename"] = (
            "/wise_output/scen0/perim.kml"
        )
        
        data_in["project"]["stations"]["wxStationData"][0]["streams"][0]["condition"]["filename"] = (
            weather_path
        )
        
        data_in["project"]["grid"]["fuelMap"]["filename"] = (
            spat_path+"fuel.asc"
        )
        data_in["project"]["grid"]["elevation"]["filename"] = (
            spat_path+"dem.asc"
        )
        data_in["project"]["grid"]["projection"]["filename"] = (
            spat_path+"dem.prj"
        )
               
        replace_in_dict(data_in, "scen0", scen_name + "_" + ignition_start[0:10])

        with open(job_name, "w") as f:
            json.dump(data_in, f, indent=2)
        print("fgmj file modified")

    # current date for filename
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M")

    # List of scenario names
    scen_names = [
        scen_name_1, scen_name_2, scen_name_3,
        scen_name_4, scen_name_5, scen_name_6,
        scen_name_7, scen_name_8, scen_name_9
    ]


    # Corresponding weather paths
    weather_paths = [f"/wise_output/area{i}/Inputs/weather.txt" for i in range(1, 10)]

    # Corresponding spat paths
    spat_paths = [f"/testjobs/testjobs/area{i}/Inputs/" for i in range(1, 10)]

    # Corresponding job filenames
    job_files = [f"/wise_output/area{i}/job.fgmj" for i in range(1, 10)]

    # Corresponding ignition coordinates
    ignition_x = [ignition_x_1, ignition_x_2, ignition_x_3,
                ignition_x_4, ignition_x_5, ignition_x_6,
                ignition_x_7, ignition_x_8, ignition_x_9]

    ignition_y = [ignition_y_1, ignition_y_2, ignition_y_3,
                ignition_y_4, ignition_y_5, ignition_y_6,
                ignition_y_7, ignition_y_8, ignition_y_9]

    # Append formatted datetime to all scenario names
    scen_names = [name + "_" + str(formatted_datetime) for name in scen_names]

    # Create job files
    for i in range(9):
        if fwi_starting_codes is not None and fwi_starting_codes[i] is None:
            continue
        create_job(
            fgmj_data_list[i],
            job_files[i],
            scen_names[i],
            ignition_x[i],
            ignition_y[i],
            weather_paths[i],
            spat_paths[i],
            fwi_starting_codes[i] if fwi_starting_codes is not None else None,
            i,
        )

    # Append ignition start date to scenario names to match the folder names created by WISE
    scen_names = [name + "_" + ignition_start[:10] for name in scen_names]

    # Print done message
    print("modify_fgmj.py done")
    
    return scen_names[0], scen_names[1], scen_names[2], scen_names[3], scen_names[4], scen_names[5], scen_names[6], scen_names[7], scen_names[8] 


def ncdf_edits_multiarea(
    dataset_path, fwi_file=None, simulation_date=None, fwi_diagnostics=None,
    scenario_start_hour=10, scenario_end_hour=21,
):
    print("running ncdf_edits_multiarea.py")
    # load netcdf dataset
    dataset = xr.open_dataset(dataset_path)

    # calculate wind speed and direction from 10u and 10v components
    wind_speed = np.sqrt(dataset["10u"] ** 2 + dataset["10v"] ** 2)
    dataset["wind_speed"] = wind_speed

    wind_direction_rad = np.arctan2(dataset["10v"], dataset["10u"])
    wind_direction_deg = np.degrees(wind_direction_rad)
    wind_direction_deg = (wind_direction_deg + 360) % 360
    dataset["wind_direction"] = wind_direction_deg

    # calculate relative humidity and convert temperatures to Celsius
    temperature_celsius = dataset["2t"] - 273.15  # Convert from Kelvin to Celsius
    dewpoint_celsius = dataset["2d"] - 273.15  # Convert from Kelvin to Celsius
    relative_humidity = 100 * (
        np.exp((17.625 * dewpoint_celsius) / (243.04 + dewpoint_celsius))
        / np.exp((17.625 * temperature_celsius) / (243.04 + temperature_celsius))
    )

    dataset["relative_humidity"] = relative_humidity
    dataset["temperature"] = temperature_celsius

    area_coords = AREA_COORDS
    (
        (area1_lat, area1_lon), (area2_lat, area2_lon), (area3_lat, area3_lon),
        (area4_lat, area4_lon), (area5_lat, area5_lon), (area6_lat, area6_lon),
        (area7_lat, area7_lon), (area8_lat, area8_lon), (area9_lat, area9_lon),
    ) = area_coords
    
    dataframes = []
    for lat, lon in area_coords:
        lon_index = int(np.argmin(np.abs((dataset.lon.values - lon + 180) % 360 - 180)))
        nearest_cell = dataset.sel(lat=lat, method="nearest").isel(lon=lon_index)
        dataframes.append(nearest_cell.to_dataframe())


    utc_times = pd.DatetimeIndex(dataset.time.values).tz_localize("UTC")
    if (
        len(utc_times) < 2
        or utc_times.hasnans
        or not (np.diff(utc_times.asi8) == pd.Timedelta(hours=1).value).all()
        or not (utc_times == utc_times.floor("h")).all()
    ):
        raise ValueError("Climate weather must have consecutive, unique whole-hour UTC timestamps")
    run_date = pd.Timestamp(
        simulation_date if simulation_date is not None else utc_times[0].tz_localize(None)
    ).normalize()
    end_date = utc_times[-1].tz_localize(None).normalize()
    if not (0 <= scenario_start_hour <= 23 and 0 <= scenario_end_hour <= 23):
        raise ValueError("Scenario hours must be between 0 and 23")
    ignition_start = run_date + pd.Timedelta(hours=scenario_start_hour)
    scenario_end = end_date + pd.Timedelta(hours=scenario_end_hour)
    if scenario_end <= ignition_start:
        raise ValueError("Scenario end must be after ignition")

    variables_to_drop = ["10v", "10u", "2t", "2d"]
    weather_starts = []
    for i, df in enumerate(dataframes):
        code, zone_id, fixed_zone, zone_name = run_timezone(run_date, i, scenario_start_hour)
        local_times = utc_times.tz_convert(fixed_zone)
        start = ignition_start.tz_localize(fixed_zone)
        end = scenario_end.tz_localize(fixed_zone)
        if start < local_times[0] or end > local_times[-1]:
            raise ValueError(f"area{i + 1}: weather does not cover the local scenario")
        scenario_times = pd.date_range(start, end, freq="h").tz_convert(ZoneInfo(zone_name))
        if any(time.utcoffset() != start.utcoffset() for time in scenario_times):
            raise ValueError(
                f"area{i + 1}: scenario crosses a timezone offset change in {zone_name}; "
                "split the run into separate dates"
            )
        civil_times = utc_times.tz_convert(ZoneInfo(zone_name))
        if any(time.utcoffset() != start.utcoffset() for time in civil_times):
            print(
                f"WARNING: area{i + 1}: weather spans an offset change in {zone_name}; "
                f"all rows use the fixed {code} offset selected at local ignition. "
                "UTC instants are preserved; pre/post-transition clock labels may differ "
                "from civil time outside the scenario."
            )
        weather_starts.append(local_times[0].isoformat())
        print(
            f"area{i + 1}: timezone={code}, timezoneId={zone_id}, "
            f"weather={local_times[0].isoformat()} to {local_times[-1].isoformat()}, "
            f"scenario={start.isoformat()} to {end.isoformat()}"
        )
        df.reset_index(inplace=True)
        df["time"] = local_times
        df.set_index("time", inplace=True)
        df["date"] = df.index.date
        df["hour"] = df.index.time
        dataframes[i] = df.drop(variables_to_drop, axis=1)

    fwi_starting_codes = None
    if fwi_file is not None:
        fwi_starting_codes = read_fwi_starting_codes(
            fwi_file,
            run_date,
            area_coords,
            diagnostics=fwi_diagnostics,
        )
    else:
        print("No FWI file supplied; retaining template FFMC, DMC and DC starting codes.")

    # set column order and rename
    new_column_order = ["date", "hour", "temperature", "relative_humidity", "wind_direction", "wind_speed", "tp"]
    column_rename_map = {
        "date": "HOURLY",
        "hour": "HOUR",
        "temperature": "TEMP",
        "relative_humidity": "RH",
        "wind_direction": "WD",
        "wind_speed": "WS",
        "tp": "PRECIP",
    }
    
    for i in range(len(dataframes)):
        dataframes[i].reset_index(inplace=True)
        dataframes[i] = dataframes[i][new_column_order].copy()
        dataframes[i].rename(columns=column_rename_map, inplace=True)
        dataframes[i]["HOURLY"] = pd.to_datetime(dataframes[i]["HOURLY"], format="%d/%m/%Y")
        dataframes[i]["HOUR"] = dataframes[i]["HOUR"].apply(lambda x: x.hour).astype(int)
        weather_columns = ["TEMP", "RH", "WD", "WS", "PRECIP"]
        dataframes[i][weather_columns] = dataframes[i][weather_columns].round(1)
        dataframes[i]["HOURLY"] = dataframes[i]["HOURLY"].dt.strftime("%d/%m/%Y")

    # save the new .txt format weather files to their designated job folders for WISE runs
    file_path = "/wise_output/"
    
    # Create directories and save weather files
    for i in range(1, 10):
        directory = f"/wise_output/area{i}/Inputs"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        file_name = f"{file_path}area{i}/Inputs/weather.txt"
        dataframes[i-1].to_csv(file_name, sep=",", index=False)


    # current working dir
    current_directory = os.getcwd()

    # run the modify_fgmj.py script
    print("ncdf_edits_multiarea.py done, starting modify_fgmj.py")
    scen_name_1,scen_name_2,scen_name_3,scen_name_4,scen_name_5,scen_name_6,scen_name_7,scen_name_8,scen_name_9 = modify_fgmj(
        utc_times[0].tz_localize(None).isoformat(),
        scenario_end.isoformat(),
        ignition_start.isoformat(),
        scenario_end.isoformat(),
        area1_lat,
        area1_lon,
        area2_lat,
        area2_lon,
        area3_lat,
        area3_lon,
        area4_lat,
        area4_lon,
        area5_lat,
        area5_lon,
        area6_lat,
        area6_lon,
        area7_lat,
        area7_lon,
        area8_lat,
        area8_lon,
        area9_lat,
        area9_lon,
        fwi_starting_codes=fwi_starting_codes,
        weather_starts=weather_starts,

    )
    return scen_name_1,scen_name_2,scen_name_3,scen_name_4,scen_name_5,scen_name_6,scen_name_7,scen_name_8,scen_name_9
    
    


def run_wise(
    in_path, out_path, year_start, month_start, day_start, year_end, month_end, day_end,
    fwi_file=None, fwi_dir=None, scenario_start_hour=10, scenario_end_hour=21,
):
    print("running run_wise.py")
    simulation_date = np.datetime64(
        datetime(int(year_start), int(month_start), int(day_start)), "ns"
    )
    fwi_file = resolve_fwi_file(simulation_date, fwi_file=fwi_file, fwi_dir=fwi_dir)
    if fwi_dir is not None:
        print(f"FWI directory={fwi_dir}; selected previous-day component file={fwi_file}")
    # Provide the data file name for all variables (weekly)
    temp_name = f"{year_start}_{month_start}_{day_start}_T00_to_{year_end}_{month_end}_{day_end}_T23_2t_timestep_60_hourly_mean.nc"  # temperature
    dewpoint_name = f"{year_start}_{month_start}_{day_start}_T00_to_{year_end}_{month_end}_{day_end}_T23_2d_timestep_60_hourly_mean.nc"  # dewpoint temperature
    uwind_name = f"{year_start}_{month_start}_{day_start}_T00_to_{year_end}_{month_end}_{day_end}_T23_10u_timestep_60_hourly_mean.nc"  # u wind
    vwind_name = f"{year_start}_{month_start}_{day_start}_T00_to_{year_end}_{month_end}_{day_end}_T23_10v_timestep_60_hourly_mean.nc"  # v wind
    precip_name = f"{year_start}_{month_start}_{day_start}_T00_00_to_{year_end}_{month_end}_{day_end}_T23_00_avg_tprate_raw_data.nc"  # precipitation


    required_files = [
        temp_name,
        dewpoint_name,
        uwind_name,
        vwind_name,
        precip_name,
    ]

    for filename in required_files:
        full_path = os.path.join(in_path, filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Required climate file missing: {full_path}"
            )


    # read the netcdf files and take variables
    temp_nc = xr.open_dataset(os.path.join(in_path, temp_name))
    dewpoint_nc = xr.open_dataset(os.path.join(in_path, dewpoint_name))
    windu_nc = xr.open_dataset(os.path.join(in_path, uwind_name))
    windv_nc = xr.open_dataset(os.path.join(in_path, vwind_name))
    precip_nc = xr.open_dataset(os.path.join(in_path, precip_name))

    windu_var = windu_nc["10u"]
    windv_var = windv_nc["10v"]
    temp_var = temp_nc["2t"]
    dewpoint_var = dewpoint_nc["2d"]
    precip_var = precip_nc["avg_tprate"] * 3600

    # combine all variables into singular file
    combined_nc = xr.Dataset(
        {
            "10u": windu_var,
            "10v": windv_var,
            "2t": temp_var,
            "2d": dewpoint_var,
            "tp": precip_var,
        }
    )

    file_name = os.path.join(in_path, "combined_ncdf.nc")

    # write the new netcdf file
    combined_nc.to_netcdf(file_name)

    # current working dir
    current_directory = os.getcwd()

    # run the ncdf_edits_multiarea.py script
    fwi_diagnostics = {"status": "template", "warnings": [], "areas": {}}
    scen_name_1, scen_name_2, scen_name_3, scen_name_4, scen_name_5, scen_name_6, scen_name_7, scen_name_8, scen_name_9 = ncdf_edits_multiarea(
        file_name, fwi_file=fwi_file, simulation_date=simulation_date,
        fwi_diagnostics=fwi_diagnostics,
        scenario_start_hour=scenario_start_hour, scenario_end_hour=scenario_end_hour,
    )

    # run the WISE model for all test areas
    print("launching WISE runs")

    scen_names = [
        scen_name_1, scen_name_2, scen_name_3,
        scen_name_4, scen_name_5, scen_name_6,
        scen_name_7, scen_name_8, scen_name_9,
    ]
    base_path = out_path
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    status_path = os.path.join(
        base_path, f"{year_start}_{month_start}_{day_start}_{run_id}_status.json"
    )
    log_path = os.path.join(base_path, f"wise_logs_{run_id}")
    os.makedirs(log_path, exist_ok=False)
    report = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "date_start": f"{year_start}-{month_start}-{day_start}",
        "date_end": f"{year_end}-{month_end}-{day_end}",
        "timeout_seconds": WISE_TIMEOUT,
        "fwi_file": os.path.abspath(fwi_file) if fwi_file is not None else None,
        "fwi_directory": os.path.abspath(fwi_dir) if fwi_dir is not None else None,
        "fwi_initialization": fwi_diagnostics,
        "areas": [
            {
                "area": i,
                "scenario": scen_names[i - 1],
                "status": "pending",
                "return_code": None,
                "outputs": {},
                "stdout_log": os.path.join(log_path, f"area{i}.stdout.log"),
                "stderr_log": os.path.join(log_path, f"area{i}.stderr.log"),
            }
            for i in range(1, 10)
        ],
    }

    def save_report():
        with open(status_path + ".tmp", "w") as handle:
            json.dump(report, handle, indent=2)
        os.replace(status_path + ".tmp", status_path)

    def save_logs(area, stdout, stderr):
        for key, content in [("stdout_log", stdout), ("stderr_log", stderr)]:
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            with open(area[key], "w") as handle:
                handle.write(content or "")

    save_report()
    print(f"Run status report: {status_path}")
    successful_areas = []

    for i in range(1, 10):
        area = report["areas"][i - 1]
        fwi_area = fwi_diagnostics["areas"].get(i, {})
        if fwi_area.get("status") == "invalid_values":
            area["status"] = "fwi_skipped"
            area["error"] = fwi_area["error"]
            save_report()
            continue
        area["status"] = "running"
        area["started_at"] = datetime.now().isoformat()
        save_report()
        stdout = ""
        stderr = ""
        cmd = [
            "wise",
            "-r",
            "4",
            "-f",
            "0",
            "-t",
            f"/wise_output/area{i}/job.fgmj",
        ]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=WISE_TIMEOUT,
            )
            area["return_code"] = result.returncode
            stdout, stderr = result.stdout, result.stderr

            if result.returncode != 0:
                area["status"] = "model_failed"
                area["error"] = f"WISE exited with return code {result.returncode}"
                print(
                    f"WISE failed for area{i}\n"
                    f"Return code: {result.returncode}"
                )

                print(result.stderr)

                continue

            successful_areas.append(i)
            area["status"] = "model_succeeded"

        except subprocess.TimeoutExpired as error:
            area["status"] = "timeout"
            area["error"] = f"WISE exceeded {WISE_TIMEOUT} seconds"
            stdout, stderr = error.stdout, error.stderr
            print(
                f"WISE timeout for area{i} "
                f"(exceeded {WISE_TIMEOUT} seconds)"
            )

            continue

        except OSError as e:
            area["status"] = "launch_failed"
            area["error"] = str(e)
            stderr = str(e)
            print(
                f"Unexpected WISE error for area{i}: "
                f"{e}"
            )

            continue
        finally:
            area["finished_at"] = datetime.now().isoformat()
            save_logs(area, stdout, stderr)
            save_report()


    print("WISE runs complete")

    print(
        f"WISE completed for "
        f"{len(successful_areas)} / 9 areas"
    )

    if len(successful_areas) < 9:

        failed = [
            i
            for i in range(1, 10)
            if i not in successful_areas
        ]

        print(
            f"Failed areas: {failed}"
        )

    if not successful_areas:
        print(
            "All WISE areas failed for this date. "
            "Recording failure and continuing without NetCDF outputs."
        )

    print("Creating NetCDF outputs")

    raster_names = [
        "burn_grid",
        "crown_fuel_consumed",
        "max_crown_fraction_burned",
        "max_flame_length",
        "max_intensity",
        "surface_fuel_consumed",
        "total_fuel_consumption",
    ]

    for i in successful_areas:
        area = report["areas"][i - 1]
        scen_name = area["scenario"]
        area["status"] = "converting"
        save_report()

        missing_files = []

        for raster_name in raster_names:

            raster_path = os.path.join(
                base_path,
                f"{scen_name}/{raster_name}.tif"
            )

            if not os.path.exists(raster_path):
                missing_files.append(raster_name)
                area["outputs"][raster_name] = {"status": "missing"}

        if missing_files:
            area["status"] = "missing_outputs"
            area["error"] = f"Missing outputs: {missing_files}"
            for raster_name in raster_names:
                if raster_name not in missing_files:
                    area["outputs"][raster_name] = {"status": "skipped"}
            save_report()
            print(
                f"Skipping {scen_name}. "
                f"Missing outputs: {missing_files}"
            )

            continue

        for raster_name in raster_names:

            raster_path = os.path.join(
                base_path,
                f"{scen_name}/{raster_name}.tif"
            )

            netcdf_path = os.path.join(
                base_path,
                f"{scen_name}/{year_start}_{month_start}_{day_start}_T00_00_{raster_name}.nc"
            )

            try:

                with rasterio.open(raster_path) as src:

                    raster_data = src.read(1)

                    print(
                        f"{raster_name}: CRS={src.crs}, "
                        f"size={src.width}x{src.height}"
                    )

                    transform, width, height = calculate_default_transform(
                        src.crs,
                        "EPSG:4326",
                        src.width,
                        src.height,
                        *src.bounds
                    )

                    if raster_name == "burn_grid":
                        resampling_method = Resampling.nearest
                    else:
                        resampling_method = Resampling.bilinear

                    dst = np.empty(
                        (height, width),
                        dtype=raster_data.dtype
                    )

                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dst,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=resampling_method,
                    )

                    lon = np.array([
                        xy(transform, 0, c, offset="center")[0]
                        for c in range(width)
                    ])

                    lat = np.array([
                        xy(transform, r, 0, offset="center")[1]
                        for r in range(height)
                    ])

                    ds = xr.Dataset(
                        {
                            raster_name: (
                                ["time", "lat", "lon"],
                                dst[np.newaxis, :, :],
                            ),
                        },
                        coords={
                            "time": [simulation_date],
                            "lat": lat,
                            "lon": lon,
                        },
                    )

                    ds["spatial_ref"] = xr.DataArray(0)

                    crs_wkt = rasterio.crs.CRS.from_epsg(4326).to_wkt()
                    ds["spatial_ref"].attrs = {
                        "grid_mapping_name": "latitude_longitude",
                        "epsg_code": "EPSG:4326",
                        "semi_major_axis": 6378137.0,
                        "inverse_flattening": 298.257223563,
                        "longitude_of_prime_meridian": 0.0,
                        "crs_wkt": crs_wkt,
                        "spatial_ref": crs_wkt,
                        "GeoTransform": " ".join(
                            str(value) for value in transform.to_gdal()
                        ),
                    }

                    ds["lat"].attrs = {
                        "standard_name": "latitude",
                        "long_name": "latitude",
                        "units": "degrees_north",
                        "axis": "Y",
                    }
                    ds["lon"].attrs = {
                        "standard_name": "longitude",
                        "long_name": "longitude",
                        "units": "degrees_east",
                        "axis": "X",
                    }
                    ds["time"].attrs = {
                        "standard_name": "time",
                        "long_name": "simulation date",
                        "axis": "T",
                        "comment": (
                            "Simulation start date at 00:00, used as a daily label; "
                            "not the raster export timestamp or an hourly snapshot."
                        ),
                    }
                    ds["time"].encoding = {
                        "units": "days since 1990-01-01 00:00:00",
                        "calendar": "standard",
                        "dtype": "float64",
                        "_FillValue": None,
                    }
                    ds["lat"].encoding["_FillValue"] = None
                    ds["lon"].encoding["_FillValue"] = None
                    ds["spatial_ref"].encoding["_FillValue"] = None

                    raster_units = {
                        "burn_grid": "1",
                        "crown_fuel_consumed": "kg m-2",
                        "max_crown_fraction_burned": "1",
                        "max_flame_length": "m",
                        "max_intensity": "kW m-1",
                        "surface_fuel_consumed": "kg m-2",
                        "total_fuel_consumption": "kg m-2",
                    }
                    ds[raster_name].attrs["units"] = raster_units[raster_name]
                    if raster_name == "burn_grid":
                        ds[raster_name].attrs["flag_values"] = np.array(
                            [0, 1], dtype=dst.dtype
                        )
                        ds[raster_name].attrs["flag_meanings"] = "non_burned burned"
                    ds[raster_name].attrs["description"] = (
                        f"{raster_name} from Raster"
                    )
                    if raster_name == "max_crown_fraction_burned":
                        ds[raster_name].attrs["description"] = (
                            "Maximum crown fraction burned, expressed as a fraction "
                            "from 0 to 1 (not percent)"
                        )
                    ds[raster_name].attrs["grid_mapping"] = "spatial_ref"
                    if src.nodata is not None:
                        ds[raster_name].encoding["_FillValue"] = src.nodata

                    ds.attrs["Conventions"] = "CF-1.8"
                    ds.attrs["crs"] = "EPSG:4326"
                    ds.attrs["scenario"] = scen_name
                    ds.attrs["source_crs_wkt"] = src.crs.to_wkt()
                    ds.attrs["resampling"] = resampling_method.name

                    for attr in [
                        "activity",
                        "dataset",
                        "experiment",
                        "generation",
                        "type",
                        "levtype",
                        "model",
                        "class",
                        "realization",
                        "stream",
                        "resolution",
                        "expver",
                    ]:

                        if attr in temp_nc.attrs:
                            ds.attrs[attr] = temp_nc.attrs[attr]

                    ds.attrs["history"] = (
                        "Created on "
                        + pd.Timestamp.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                    ds.to_netcdf(netcdf_path)
                    area["outputs"][raster_name] = {
                        "status": "success",
                        "path": netcdf_path,
                    }

                    print(
                        f"NetCDF file created: "
                        f"{netcdf_path}"
                    )

            except Exception as e:
                area["outputs"][raster_name] = {
                    "status": "conversion_failed",
                    "error": str(e),
                }
                print(
                    f"Failed to create NetCDF "
                    f"for {raster_path}: {e}"
                )

                continue
            finally:
                save_report()

        area["status"] = (
            "success"
            if all(
                output["status"] == "success"
                for output in area["outputs"].values()
            )
            else "conversion_failed"
        )
        save_report()

    complete = sum(area["status"] == "success" for area in report["areas"])
    report["status"] = (
        "success" if complete == 9 else "partial" if complete else "failed"
    )
    report["complete_areas"] = complete
    report["finished_at"] = datetime.now().isoformat()
    save_report()
    print(
        f"Date output status: {report['status']}; "
        f"{complete} / 9 areas have complete NetCDF outputs. "
        f"Details: {status_path}"
    )
    
# === Restart File Handling (Only on the 1st of the Month) ===
    if int(day_start) == 1:
        restart_file_path = os.path.join(base_path, f"{year_start}_{month_start}_{day_start}_T00_00_restart_file.nc")
        prev_month = str(int(month_start) - 1).zfill(2)
        prev_year = year_start if month_start != "01" else str(int(year_start) - 1)
        prev_restart_file = os.path.join(base_path, f"{prev_year}_{prev_month}_{day_start}_T00_00_restart_file.nc")

        # Create an empty restart file
        restart_ds = xr.Dataset(attrs={"description": "Restart file for WISE runs"})
        restart_ds.to_netcdf(restart_file_path)
        print(f"Restart file created: {restart_file_path}")

        # Delete the previous restart file if it exists
        if os.path.exists(prev_restart_file):
            os.remove(prev_restart_file)
            print(f"Deleted previous restart file: {prev_restart_file}")


def main():
    # defining file input / output paths
    in_path = "/input_data/"
    
    parser = argparse.ArgumentParser(description="Runscript for data notifier job.")

    parser.add_argument(
        "-year_start", required=True, help="Input year start", default=1
    )
    parser.add_argument(
        "-month_start", required=True, help="Input month start", default=2
    )
    parser.add_argument("-day_start", required=True, help="Input day start", default=3)

    parser.add_argument("-year_end", required=True, help="Input year end", default=4)
    parser.add_argument("-month_end", required=True, help="Input month end", default=5)
    parser.add_argument("-day_end", required=True, help="Input day end", default=6)
    fwi_inputs = parser.add_mutually_exclusive_group()
    fwi_inputs.add_argument(
        "--fwi-file",
        help=(
            "FWI component NetCDF containing FFMC, DMC and DC for the calendar "
            "day before the simulation start date. "
            "If omitted, retain template starting codes."
        ),
    )
    fwi_inputs.add_argument(
        "--fwi-dir",
        help=(
            "FWI component directory. Select YYYY_MM_DD_fwi_comp.nc for the "
            "calendar day before the simulation start date."
        ),
    )

    # parser.add_argument('-expid', required=True, help='experiment id', default=7)

    args = parser.parse_args()

    out_path = "/wise_output/"

    run_wise(
        in_path,
        out_path,
        args.year_start,
        args.month_start,
        args.day_start,
        args.year_end,
        args.month_end,
        args.day_end,
        fwi_file=args.fwi_file,
        fwi_dir=args.fwi_dir,
    )


if __name__ == "__main__":
    main()
