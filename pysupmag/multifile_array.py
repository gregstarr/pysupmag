import numpy as np
import xarray as xr


class MultifileBaseClass:
    """
    This will fill in for an nd-array but will span multiple files on disk. This is essentially one giant
    array, except under the hood, it will open up different files depending on which elements are accessed.

    In order to make this more flexible, i.e. work with xarray DataArrays, DataSets, Numpy arrays, pandas etc. this
    will be subclassed into specific under-the-hood types through the _load_cache function.

    The _load_cache function needs to load up a file and turn it into a big numpy array, then the _item_handler
    needs to take the item request (indexing / slices) and turn it into something that you would give to a numpy
    array
    """
    def __init__(self, file_n, file_idx, paths, shape):
        self.file_n = file_n
        self.file_idx = file_idx
        self.paths = paths
        self.shape = shape
        if not isinstance(self.shape, tuple):
            self.shape = tuple(self.shape)
        self.cache = None
        self.file_num_in_cache = None

    def _load_cache(self, file_n):
        """This needs to be overridden by the subclass.
        """
        pass

    def _item_handler(self, item):
        """This needs to be overridden by the subclass.
        """
        return item

    def _slice_to_idx(self, s):
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else self.file_n.shape[0]
        step = s.step if s.step is not None else 1
        return np.arange(start, stop, step)

    def _process_request(self, item):
        # first item detemines slice 1
        shape1 = None
        if not isinstance(item, tuple):
            item = (item, )
        slice2 = item[1:] or (slice(None), )
        if isinstance(item[0], int):
            slice1 = (np.array([item[0]]), )
        elif isinstance(item[0], slice):
            slice1 = (self._slice_to_idx(item[0]), )
        elif isinstance(item[0], np.ndarray):
            # mask can decrease dimension
            if item[0].dtype == bool:
                idx = np.argwhere(item[0])
                slice1 = tuple([idx[:, j] for j in range(idx.shape[1])])
            elif item[0].dtype == int:
                shape1 = item[0].shape
                slice1 = (item[0].ravel(), )
        return slice1, slice2, shape1

    def __getitem__(self, item):
        """This is not elegant.
        """
        # convert the given item into a numpy slice
        item = self._item_handler(item)
        # virtual slice 1, virtual slice 2, and reshape for first dimension if necessary
        vslice1, vslice2, shape1 = self._process_request(item)
        # determine required files
        files_to_access = np.unique(self.file_n[vslice1[0]])
        # initialize list for data from different files
        requested_data = [None for _ in files_to_access]
        # TODO: check if the currently loaded cache can be used first, then remove that file from files_to_access
        # iterate through the files
        for j, file_n in enumerate(files_to_access):
            # load the current file
            self._load_cache(file_n)
            # which elements of the request are found in this file?
            request_in_file_mask = self.file_n[vslice1[0]] == file_n
            # file slice 1 needs to be restricted to this file
            fslice1 = (self.file_idx[vslice1[0][request_in_file_mask]], )
            for i in range(1, len(vslice1)):
                fslice1 += (vslice1[i][request_in_file_mask], )
            # second fslice are elements that all files have in the same place
            fslice2 = (slice(None), ) + vslice2
            # re assign list element to the sliced data
            requested_data[j] = self.cache[fslice1][fslice2]
        # concatenate the list together
        if len(requested_data) > 1:
            requested_data = np.concatenate(requested_data, axis=0)
        else:
            requested_data = requested_data[0]
        # reshape the first dimenion if necessary
        if shape1 is not None:
            return requested_data.reshape(shape1 + requested_data.shape[1:])
        return requested_data


class MultifileNumpyArray(MultifileBaseClass):

    def _load_cache(self, file_n):
        self.cache = np.load(self.paths[file_n])
        self.file_num_in_cache = file_n


class MultifileXarray(MultifileBaseClass):
    """
    This is made more complicated by the fact that each dataset could have a different set of stations in it.
    Options:
        - ensure ahead of time that all the mag files have the same stations (extent with NaNs)
        - initialize this object with the list of stations which constitute the overall 'array'  <--  favorite
        - access with a list of stations
    """
    def __init__(self, file_n, file_idx, paths, stations):
        self.stations = stations
        self.station_len = len(self.stations)
        super().__init__(file_n, file_idx, paths, (self.station_len, 5))

    def _load_cache_xarray(self, file_n):
        time_len = np.sum(self.file_n == file_n)
        self.cache = np.ones((time_len,) + self.shape) * np.nan  # (time x station x component)
        data = xr.open_dataset(self.paths[file_n]).sel(dim_1=['MLT', 'MLAT', 'N', 'E', 'Z'])
        for i, st in enumerate(self.stations):
            if st in data:
                self.cache[:, i] = data[st].values
        self.file_num_in_cache = file_n
