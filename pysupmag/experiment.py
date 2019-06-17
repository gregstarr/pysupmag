import glob
import pandas as pd
import numpy as np
from pysupmag.dataset import DataSource, DataCollection

DATA_DIR = "C:/Users/Greg/code/substorm-detection/data/"

print("Solar Wind")
solar_wind_fn = DATA_DIR + "solar_wind.pkl"
data = pd.read_pickle(solar_wind_fn)
sw = DataSource("solar_wind", data.values, data.index)

print("SME - filename")
sme_fn = DATA_DIR + "SME.csv"
print("SME - read csv")
data = pd.read_csv(sme_fn, index_col=0)
print("SME - data source")
sme = DataSource("sme", data.values, pd.to_datetime(data.index))

print("Substorms - filename")
substorm_fn = DATA_DIR + "substorms.csv"
print("Substorms - read csv")
data = pd.read_csv(substorm_fn)
print("Substorms - to datetime")
data.index = pd.to_datetime(data['Date_UTC'])
print("Substorms - drop")
data = data.drop(columns=['Unnamed: 0', 'Date_UTC'])
print("Substorms - datasource")
substorms = DataSource("substorms", data.values, data.index)

print("Mag")
paths = glob.glob(DATA_DIR + "mag_data/mag_data*.nc")
mag = DataSource.from_xarray_files("mag", paths)

print("Collection")
collection = DataCollection(sources=[mag, sw, sme, substorms])

# TODO: make regression dataset
Tm = 100
Tw = 100
Tsme = 20
# number of examples should be 1 or 2 per substorm
# randomly select that many datetime indices from master list
print("randomly select times")
example_date_idx = np.random.choice(np.arange(collection.dates.shape[0] // 10, dtype=int), 1000, replace=False)

# grab corresponding data and targets
# mag data is from t0 - Tm : t0
print("mag data")
mag_data, mag_mask = mag.get_data(example_date_idx, before=Tm)

# solar wind data is from t0 - Tw : t0
print("sw data")
sw_data, sw_mask = sw.get_data(example_date_idx, before=Tw)

# target is t_next_substorm - t0
print("targets")
ss_idx = substorms.get_next(example_date_idx)  # returns dates? or indices?
targets = ss_idx - example_date_idx

# sme is lowest sml over t_next_substorm : t_next_substorm + Tsme
print("sme")
sme_data, sme_mask = sme.get_data(ss_idx, after=Tsme)
sme_data = sme_data[:, :, 1].min(axis=1)

# location
# mask out examples with bad data or outside of region

print()
