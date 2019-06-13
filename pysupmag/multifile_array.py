import numpy as np
import xarray as xr


class MultifileArray:
    """
    This will fill in for a regular numpy array but will span multiple files on disk. This is essentially one giant
    array, except under the hood, it will open up different files depending on which elements are accessed. This is
    made more complicated by the fact that each dataset could have a different set of stations in it.
    Options:
        - ensure ahead of time that all the mag files have the same stations (extent with NaNs)
        - initialize this object with the list of stations which constitute the overall 'array'  <--  favorite
        - access with a list of stations

    In order to make this more flexible, i.e. work with xarray DataArrays, DataSets, Numpy arrays, pandas etc. consider
    having the data loading function be configured by the user so it can open up no matter what. The user provided
    function would accept the file number and would return the data, time axis first, as a numpy array.
    """
    def __init__(self, file_map, paths, stations):
        """

        Parameters
        ----------
        file_map: numpy.ndarray (T x 2) - first column is index into this data source, second column is the
            corresponding file number
        paths: list - file paths
        stations: list - string station names
        """
        self.file_map = file_map  # file_map[index] = [file number, index within that file]
        self.paths = paths
        self.stations = stations
        self.station_len = len(self.stations)
        self.cache = None
        self.file_num_in_cache = None

    def _load_cache(self, file_n):
        time_len = np.sum(self.file_map[:, 0] == file_n)
        self.cache = np.ones((time_len, self.station_len, 5)) * np.nan  # (time x station x component)
        data = xr.open_dataset(self.paths[file_n]).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
        for i, st in enumerate(self.stations):
            if st in data:
                self.cache[:, i] = data[st].values
        self.file_num_in_cache = file_n

    def _get_single_item(self, item):
        if self.file_num_in_cache != self.file_map[item, 0]:
            self._load_cache(self.file_map[item, 0])
        return self.cache[self.file_map[item, 1]]

    def _parse_slice(self, s):
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else self.file_map[:, 1].max() + 1
        step = s.step if s.step is not None else 1
        return np.arange(start, stop, step)

    def __getitem__(self, item):
        """This needs to work with int, slice, boolean array, int array and tuples of any combination
        'item' would slice a properly sized array, I just need to turn the first tuple element into a numpy
        array and break up accordingly. The other tuple elements distribute.
        """
        # determine if slice or single element
        if isinstance(item, int):
            return self._get_single_item(item)
        elif isinstance(item, slice):
            # initialize array I will return
            request_idx = self._parse_slice(item)
            requested_data = np.empty((request_idx.shape[0], self.station_len, 5))
            # determine required files
            files_to_access = np.unique(self.file_map[item, 0])
            # determine required indices from each file
            for file_n in files_to_access:
                self._load_cache(file_n)
                request_in_file_mask = self.file_map[request_idx, 0] == file_n
                requested_data[request_in_file_mask] = self.cache[self.file_map[request_idx[request_in_file_mask], 1]]
            return requested_data
        else:
            raise IndexError("Index must be integer or slice")