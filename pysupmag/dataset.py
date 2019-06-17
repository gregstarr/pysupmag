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


def create_regression_dataset(data_dir, n_train, n_test, Tm=128, Tw=196, Tsme=20, nan_th=.8):
    import glob

    print("create DataSource objects")
    print("Solar Wind")
    solar_wind_fn = data_dir + "solar_wind.pkl"
    data = pd.read_pickle(solar_wind_fn)
    sw = DataSource("solar_wind", data.values, data.index)

    print("SME")
    sme_fn = data_dir + "SME.csv"
    data = pd.read_csv(sme_fn, index_col=0)
    sme = DataSource("sme", data.values, pd.to_datetime(data.index))

    print("Substorms")
    substorm_fn = data_dir + "substorms.csv"
    data = pd.read_csv(substorm_fn)
    data.index = pd.to_datetime(data['Date_UTC'])
    data = data.drop(columns=['Unnamed: 0', 'Date_UTC'])
    substorms = DataSource("substorms", data.values, data.index)

    print("Mag")
    paths = glob.glob(data_dir + "mag_data/mag_data*.nc")
    mag = DataSource.from_xarray_files("mag", paths)

    print("Collection")
    collection = DataCollection(sources=[mag, sw, sme, substorms])

    # randomly select that many datetime indices from master list
    print("randomly select times")
    example_date_idx = np.sort(np.random.choice(np.arange(collection.dates.shape[0], dtype=int),
                                                n_train + n_test, replace=False))

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

    # mask out examples with missing data
    mask = mag_mask * sw_mask * sme_mask
    mag_data = mag_data[mask]
    targets = targets[mask]
    sw_data = sw_data[mask]
    sme_data = sme_data[mask]

    # figure out good ordering for the stations (rows)
    # remove stations with no data in the dataset
    station_mask = ~(np.mean(np.isnan(mag_data), axis=(0, 1, 3)) > nan_th)
    mag_data = mag_data[:, :, station_mask, :]
    print("Annealing...")
    station_locations = np.array(mag_data.data.stations)[station_mask]
    dists = utils.distance_matrix(station_locations)
    dists[np.isnan(dists)] = 0

    sa = utils.SimAnneal(station_locations, dists, stopping_iter=100000000)
    sa.anneal()
    train_dict['mag_data_train'] = train_dict['mag_data_train'][:, sa.best_solution, :, :]
    test_dict['mag_data_test'] = test_dict['mag_data_test'][:, sa.best_solution, :, :]
    stations = [self.supermag.stations[s] for s in sa.best_solution]
    station_locations = station_locations[sa.best_solution]

    data_dict = {
        'mag_data' + ext: mag_data,
        'sw_data' + ext: sw_data,
        'y' + ext: y,
        'sme_data' + ext: sme_data,
        'ss_location' + ext: ss_location,
        'ss_dates' + ext: ss_dates}

    # save the dataset
    np.savez(self.output_fn, **{**train_dict, **test_dict, 'stations': stations,
                                'station_locations': station_locations})
