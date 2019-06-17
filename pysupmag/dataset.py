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
from pysupmag.multifile_array import MultifileXarray


class DataCollection:
    """
    This will hold multiple DataSource objects, and will be responsible for serving up aligned data from each source.
    """
    def __init__(self, sources=[], timeunit='m'):
        # initialized with data source objects
        self.sources = {}
        for source in sources:
            self.sources[source.name] = source
        self.period = np.timedelta64(min([np.min(np.diff(ds.dates[:20])) for ds in self.sources.values()]), timeunit)
        self.start_date = min([ds.dates[0] for ds in self.sources.values()])
        self.end_date = max([ds.dates[-1] for ds in self.sources.values()])
        self.dates = pd.to_datetime(np.arange(self.start_date, self.end_date + self.period, self.period))
        self.update_alignment()

    def update_alignment(self):
        for source in self.sources.values():
            source.update_alignment(self.dates)

    def __getitem__(self, item):
        if item in self.sources:
            return self.sources[item]
        else:
            raise IndexError("{} not in this DataCollection".format(item))


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
        self.self_to_master = None  # same length as self.data, integer index of corresponding datetime in master index
        self.master_to_self = None

    def get_data(self, master_idx, before=0, after=0):
        """
        Collect data from indices in master, optionally include consecutive indices from before and after the
        selected ones.

        Parameters
        ----------
        master_idx: numpy.ndarray (M, ) - indices from master to collect data at
        before: int - indices before each in master_idx to include in return
        after: int - indices after each in master_idx to include in return

        Returns
        -------
        data: numpy.ndarray (after - before + 1, M, ...) data from self.data at the selected indices
        """
        idx = master_idx[:, None] + np.arange(-before, after + 1)[None, :]
        mask = np.all(np.isin(idx, self.self_to_master), axis=1)
        self_idx = self.master_to_self[idx]
        mask_expand = (slice(None), ) + (None, ) * self.data.ndim  # is this the best way of doing this?
        data = np.squeeze(np.where(mask[mask_expand], self.data[self_idx], np.nan)), mask
        return data

    def get_next(self, master_idx):
        diff = self.self_to_master[None, :] - master_idx[:, None]
        diff = np.where(diff >= 0, diff, np.inf)
        return self.self_to_master[np.argmin(diff, axis=1)]

    def update_alignment(self, master_dates):
        # assuming master_dates contains ALL datetimes and that each of self.dates is present in master_dates only once
        self.self_to_master = np.argwhere(np.in1d(master_dates, self.dates, assume_unique=True))[:, 0]
        self.master_to_self = np.ones_like(master_dates, dtype=int) * -1
        self.master_to_self[np.in1d(master_dates, self.dates, assume_unique=True)] = np.arange(self.dates.shape[0])

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
        data = MultifileXarray(file_n, file_idx, files, stations)
        dates = np.concatenate(dates)
        return cls(name, data, dates)
