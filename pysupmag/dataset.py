"""
Ultimately, the flow for creating a dataset will be:
    - figure out desired datetime indices and interval lengths for each example
    - grab full data from every source
    - mask out examples without data or outside of region

Binary Dataset:
    - grab the master datetime index of every substorm
    - DataSources will be able to return data based on this master datetime index
    - randomly choose an "interval index" or "temporal distance" for each substorm between max(0, last substorm)
        and Tp
    - grab the corresponding data (or value / target) from each data source using a gathering function from
        DataSource
    - this function will fill in missing values with NaN and provide a mask indicating where the good data is
    - substorms could additionally be masked based on region

    - randomly select datetimes which have no substorms in their corresponding prediction intervals
        - mask starts as same length of master index
        - reset all which contain substorm in next Tp min
        - randomly select dates from these
        - negative_dates = possible_dates[np.cumsum(np.random.randint(self.Tm, possible_dates.shape[0] //
            n_ss_examples, n_ss_examples))]
        - mask based on region
    - collect data same as above

Regression Dataset:
    - number of examples should be 1 or 2 per substorm
    - randomly select that many datetime indices from master list
    - grab corresponding data and targets
    - mask out examples with bad data or outside of region
"""
import numpy as np
import pandas as pd
import xarray as xr
from pysupmag.multifile_array import MultifileBaseClass


class DataCollection:
    """
    This will hold multiple DataSource objects, and will be responsible for serving up aligned data from each source.
    """
    def __init__(self, sources=[], timeunit='m'):
        # initialized with data source objects
        self.sources = sources
        self.period = np.timedelta64(min([np.min(np.diff(ds.dates[:20])) for ds in self.sources]), timeunit)
        self.start_date = min([ds.dates[0] for ds in self.sources])
        self.end_date = max([ds.dates[-1] for ds in self.sources])
        self.dates = pd.to_datetime(np.arange(self.start_date, self.end_date + self.period, self.period))
        self.update_alignment()

    def update_alignment(self):
        for source in self.sources:
            source.update_alignment(self.dates)


class DataSource:

    def __init__(self, name, data, dates):
        """

        Parameters
        ----------
        name: str - name of the data source e.g. sw, solar_wind, sme, etc.
        data: numpy.ndarray (T x D) - data from source, time axis first, data dimension second
        dates: pandas datetime index (T,) - datetime index corresponding to data
        """
        self.name = name
        self.data = data
        self.dates = dates
        self.master_to_self = None  # same length as self.data, integer index of corresponding datetime in master index

    def get_data(self, idx, before=0, after=0):
        pass

    def update_alignment(self, master_dates):
        # assuming master_dates contains ALL datetimes and that each of self.dates is present in master_dates only once
        self.master_to_self = np.argwhere(np.in1d(master_dates, self.dates))[:, 0]

    def __repr__(self):
        return "{} : {}".format(self.name, self.__class__)

    @classmethod
    def from_xarray_files(cls, name, files, stations=None):
        dates = []
        extract_stations = False
        if stations is None:
            extract_stations = True
            stations = set()
        for file in files:
            dset = xr.open_dataset(file)
            dates.append(pd.to_datetime(dset.Date_UTC.values))
            if extract_stations:
                stations = stations.union(set(s for s in dset))
        if extract_stations:
            stations = list(stations)
        file_n = np.concatenate([np.array([i] * dates[i].shape[0]) for i in range(len(dates))])
        file_idx = np.concatenate([np.arange(dates[i].shape[0]) for i in range(len(dates))])
        file_map = np.stack((file_n, file_idx), axis=1)
        data = MultifileArray(file_map, files, stations)
        dates = np.concatenate(dates)
        return cls(name, data, dates)
