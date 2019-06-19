"""
the polar plot downloads are 1 day long each
the request needs an URL and a params dictionary:
    - station data in CSV requires params
    - polar plots (fitted vectors) comes in netCDF and requires no params
    - solar wind datas
    - sme
    - substorms
"""
import requests
import xarray as xr
from io import BytesIO
import pandas as pd
import numpy as np
import threading
import queue
import time
import pickle


def process_ncdf(data):
    df = pd.DataFrame({'year': data['time_yr'], 'month': data['time_mo'], 'day': data['time_dy'],
                       'hour': data['time_hr'], 'minute': data['time_mt'], 'second': data['time_sc']})
    dates = pd.to_datetime(df, unit='m')
    db_nez = data[['dbn_nez', 'dbe_nez', 'dbz_nez']].to_array().values.transpose(1, 2, 0)
    db_nez = db_nez.reshape((-1, 24, 25, 3))
    mlt = np.arange(24)[:, None] * np.ones((1, 25))
    mlat = np.arange(88, 38, -2)[None, :] * np.ones((24, 1))

    try:
        assert np.mean(data['mlat'].values.reshape((-1, 24, 25)) == mlat[None, :, :]) > .9
    except AssertionError as e:
        print("MLAT improperly formatted")
        raise e
    try:
        assert np.mean(data['mlt'].values.reshape((-1, 24, 25)) == mlt[None, :, :]) > .9
    except AssertionError as e:
        print("MLT improperly formatted")
        raise e

    dset = xr.Dataset({'db_nez': (['time', 'x', 'y', 'component'], db_nez)},
                      coords={'mlt': (['x', 'y'], mlt),
                              'mlat': (['x', 'y'], mlat),
                              'time': pd.DatetimeIndex(dates), 'component': ['n', 'e', 'z']})
    return dset


def download_data(url_queue, results):
    while not url_queue.empty():
        i, url = url_queue.get()
        print(url)
        tries = 0
        while tries < 5:
            try:
                req = requests.get(url, timeout=120)
                if not req:
                    print("ERROR")
                    continue
                break
            except Exception as e:
                tries += 1
                print(e)
        buffer = BytesIO(req.content)
        data = xr.open_dataset(buffer)
        results[i] = process_ncdf(data)


N_THREADS = 40
DATA_DIR = "E:/mag_data_interp/"
url_format = "http://supermag.jhuapl.edu/ncdf/schavec-mlt/{year:04d}/{year:04d}{month:02d}{day:02d}.north.schavec-mlt-supermag.rev-0002.ncdf"

for year in range(1990, 2019):
    START_TIME = time.time()

    url_queue = queue.Queue(maxsize=0)
    start_date = np.datetime64("{}-01-01".format(year))
    end_date = np.datetime64("{}-01-01".format(year + 1))
    dates = pd.to_datetime(np.arange(start_date, end_date, np.timedelta64(1, 'D')))
    results = [None for _ in dates]
    for i, date in enumerate(dates):
        url_queue.put((i, url_format.format(year=date.year, month=date.month, day=date.day)))

    threads = []
    for _ in range(N_THREADS):
        thread = threading.Thread(target=download_data, args=(url_queue, results))
        thread.start()
        time.sleep(.2)
        threads.append(thread)

    for thread in threads:
        thread.join()

    dataset = xr.concat(results, dim='time')
    # get unique times
    # convert types to np.int64
    # convert types to np.int32
    # get rid of repeats
    dataset = dataset.isel(time=np.unique(dataset['time'], return_index=True)[1])
    # restrict to this year
    dataset = dataset.sel(time=str(year))
    with open(DATA_DIR + "{}.pkl".format(year), 'wb') as f:
        pickle.dump(dataset, f, protocol=-1)

    TIME_DIFF = time.time() - START_TIME
    print("Year {} took: {} minutes, {} seconds".format(year, int(TIME_DIFF // 60), int(TIME_DIFF % 60)))
