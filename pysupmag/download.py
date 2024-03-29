# -*- coding: utf-8 -*-
"""
Download data from the SuperMAG website. This is way easier than I thought! :)

I think the script will look something like this:
    - input start and end date
    - input stations / regions ???
    - go through days one at a time, checking on available stations and
      downloading data from all available stations
    - save the whole thing to a big hdf5 file, oh yeah

Created on Wed Dec 26 10:03:48 2018
"""


import requests
from datetime import datetime
import xarray as xr
import pandas as pd
from io import StringIO
import time
import os

# CONSTANTS

# download directory
DOWNLOAD_DIR = "mag_data"
N_INTERVALS = 10
START_YEAR = 1999
END_YEAR = 2000

MAX_TRIES = 5
SLEEP_TIME = 5
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.000Z"

# request string constants
GET_STATION_LIST = "http://supermag.jhuapl.edu/mag/lib/services/inventory.php"
STATION_PARAMS = {"service": "inventory",
                  "start": None,
                  "interval": None}

GET_DATA = "http://supermag.jhuapl.edu/mag/lib/services/"
DATA_PARAMS = {"?user": "gregstarr",
               "start": None,
               "interval": None,
               "service": "mag",
               "stations": None,
               "delta": "none",
               "baseline": "all",
               "options": "mlt sza decl",
               "fmt": "csv"}


def date_range(start, end, intv):
    diff = (end - start) / intv
    for i in range(intv):
        yield start + diff * i
    yield end


def getDataForStation(data_params):
    tries = 0
    while tries < MAX_TRIES:
        try:
            data_rq = requests.get(GET_DATA, data_params, timeout=10)
            break
        except:
            tries += 1
            time.sleep(SLEEP_TIME)
    if tries == MAX_TRIES:
        print("couldn't get data for {} {}".format(data_params['stations'], data_params['start']))
        return
    buffer = StringIO(data_rq.text)
    df = pd.read_csv(buffer)
    times = pd.to_datetime(df.Date_UTC)
    df.index = times
    return df.drop(columns=['Date_UTC', 'IAGA'])


def getAvailableStations(station_params):
    tries = 0
    while tries < MAX_TRIES:
        try:
            station_rq = requests.get(GET_STATION_LIST, station_params, timeout=10)
            break
        except Exception as e:
            tries += 1
            time.sleep(SLEEP_TIME)
    if tries == MAX_TRIES:
        print("couldn't get stations for {}".format(station_params['start']))
        print(e)
        return
    try:
        station_list = station_rq.json()['stations']
    except Exception as e:
        print("couldn't get stations for {}".format(station_params['start']))
        print(e)
        return
    return station_list


def getDataForInterval(start_date, hours, minutes):
    """Gets data from website for all stations over an interval

    Parameters
    ----------
    start_date: datetime
    hours: int
        number of hours in interval
    minutes: int
        minutes in interval
    Returns
    -------
    data: xarray.Dataset
        dataset with the data for all stations in the interval
    """
    data = {}

    station_params = STATION_PARAMS.copy()
    station_params['start'] = start_date.strftime(DATE_FORMAT)
    station_params['interval'] = "{}:{}".format(hours, minutes)

    # get stations for interval
    station_list = getAvailableStations(station_params)
    print("{} stations for interval starting {}".format(len(station_list),
                                                        start_date.strftime(DATE_FORMAT)))

    for station in station_list:
        data_params = DATA_PARAMS.copy()
        data_params['start'] = start_date.strftime(DATE_FORMAT)
        data_params['interval'] = "{}:{}".format(hours, minutes)
        data_params['stations'] = station

        print("{} / {}: {}".format(station_list.index(station) + 1,
                                   len(station_list), station))

        df = getDataForStation(data_params)
        data[station] = df

    return xr.Dataset(data)


def downloadDataToFile(fn, start_date, end_date, nintervals):
    dataset = xr.Dataset()

    dates = list(date_range(start_date, end_date, nintervals))
    for i in range(nintervals):
        print("Interval {} of {} for {}".format(i + 1, nintervals, start_date.year))
        tdelt = dates[i + 1] - dates[i]
        hours = int(tdelt.total_seconds() // 3600)
        minutes = int((tdelt.total_seconds() - 3600 * hours) // 60)
        data = getDataForInterval(dates[i], hours, minutes)
        dataset = xr.merge([dataset, data])

    print("saving {}".format(fn))
    dataset.to_netcdf(fn)


def downloadData():
    os.chdir(DOWNLOAD_DIR)
    for yr in range(START_YEAR, END_YEAR):
        start_date = datetime(yr, 1, 1, 0, 0, 0)
        end_date = datetime(yr + 1, 1, 1, 0, 0, 0)
        fn = "mag_data_{}".format(yr)
        downloadDataToFile(fn, start_date, end_date, N_INTERVALS)


def download_magnetometer():
    pass


def download_indices():
    pass


def download_fitted_vectors():
    pass


def download_substorms():
    pass


def generic_download_data():
    """
    - determine how many files the data should be split into: one single file / yearly file / monthly file etc.
    - for each file:
        - start N threads
        - in each thread:
            - the request (url and params) is pulled from a queue
            - the request content is turned into StringIO / BytesIO
            - the resulting IO is processed by a data-type-specific function returning an xarray DataArray / DataSet
            - this DataSet is put into the results list
        - merge the list of DataSets into one
        - save the dataset
    """
    pass


"""
In general, the way the data is downloaded is as follows:
    - determine how many files the data should be split into: one single file / yearly file / monthly file etc.
    - for each file:
        - start N threads
        - in each thread:
            - the request (url and params) is pulled from a queue
            - the request content is turned into StringIO / BytesIO
            - the resulting IO is processed by a data-type-specific function returning an xarray DataArray / DataSet
            - this DataSet is put into the results list
        - merge the list of DataSets into one
        - save the dataset
    
The datasets should share axis ordering when possible, (time, spatial dimensions, component).
It would be great to be able to save the datasets as NetCDF. 
"""


if __name__ == "__main__":
    downloadData()
