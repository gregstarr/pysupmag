import glob
import pandas as pd
import numpy as np
from data.dataset import DataSource, DataCollection

print("Solar Wind")
solar_wind_fn = "solar_wind.pkl"
data = pd.read_pickle(solar_wind_fn)
sw = DataSource("solar_wind", data.values, data.index)

print("SME - filename")
sme_fn = "SME.csv"
print("SME - read csv")
data = pd.read_csv(sme_fn, index_col=0)
print("SME - data source")
sme = DataSource("sme", data.values, pd.to_datetime(data.index))

print("Substorms - filename")
substorm_fn = "substorms.csv"
print("Substorms - read csv")
data = pd.read_csv(substorm_fn)
print("Substorms - to datetime")
data.index = pd.to_datetime(data['Date_UTC'])
print("Substorms - drop")
data = data.drop(columns=['Unnamed: 0', 'Date_UTC'])
print("Substorms - datasource")
substorms = DataSource("substorms", data.values, data.index)

print("Mag")
paths = glob.glob("mag_data/mag_data*.nc")
mag = DataSource.from_xarray_files("mag", paths)

print("Collection")
collection = DataCollection(sources=[mag, sw, sme, substorms])

print(np.isfinite(collection.sources[0].data[100_000:1_000_000]).sum())
