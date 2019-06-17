"""
Tests:
    - Numpy tests:
        (create MultifileArray, index, compare to the full array)
        - indexing:
            - integers
            - slices of the form [b:], [:e], [b:e], [b:e:s]
            - boolean masks
            - integer arrays
            - integer arrays with repeated items
            - combinations of the above on multiple dimensions !!

Create the test files (numpy arrays, etc.) in setup and pass in paths in a fixture
clean them up at the end with a fixture
"""
import pytest
import numpy as np
import xarray as xr
from tempfile import TemporaryDirectory
from pysupmag.multifile_array import MultifileXarray
import glob

N_TESTS = 100
XARRAY_FILES = glob.glob("C:/Users/Greg/code/substorm-detection/data/mag_data/mag_data*.nc")


@pytest.fixture(scope="module")
def file_map():
    lengths = np.random.randint(1, 200, 3)
    file_idx = np.concatenate([np.arange(lengths[i]) for i in range(3)]).astype(int)
    file_n = np.concatenate([np.ones(lengths[i]) * i for i in range(3)]).astype(int)
    return file_n, file_idx


@pytest.fixture(scope="module")
def xarray_data(file_map):
    file_n, file_idx = file_map
    # open up a dataset
    data = xr.open_dataset(XARRAY_FILES[0])
    # find stations with data for first length of file_n.shape[0]
    stations_with_data = [s for s in data if np.all(np.isfinite(data[s][:file_n.shape[0]]))]
    # create a dataset same length as file_n
    data = data[stations_with_data].isel(Date_UTC=slice(file_n.shape[0])).sel(dim_1=["MLT", "MLAT", "N", "E", "Z"])
    return data, stations_with_data


@pytest.fixture(scope="module")
def xarray_files(file_map, xarray_data):
    file_n, file_idx = file_map
    data, stations = xarray_data
    with TemporaryDirectory() as tempdirname:
        paths = []
        for i in range(3):
            # split dataset according to file_n, only using a subset of stations for each
            fn = "{}\\{}.nc".format(tempdirname, i)
            data.isel(Date_UTC=(file_n == i)).to_netcdf(fn)
            paths.append(fn)
        yield paths
    print("deleted tempdir")


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_xarray(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)
    check_idx = np.random.randint(file_n.shape[0])
    assert np.all(array[check_idx] == data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_slices_xarray(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    check_idx = slice(np.random.randint(1, file_n.shape[0]))
    assert np.all(array[check_idx] == data[check_idx])

    a = np.random.randint(1, file_n.shape[0] - 1)
    b = np.random.randint(1, file_n.shape[0] - a)
    check_idx = slice(a, a + b)
    assert np.all(array[check_idx] == data[check_idx])

    a = np.random.randint(10, file_n.shape[0] - 10)
    b = np.random.randint(5, file_n.shape[0] - a - 5)
    c = np.random.randint(1, b - 1)
    check_idx = slice(a, a + b, c)
    assert np.all(array[check_idx] == data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_boolean_mask_xarray(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    random_ind = np.random.randint(0, file_n.shape[0], np.random.randint(1, file_n.shape[0]))
    check_idx = np.zeros_like(file_n, dtype=bool)
    check_idx[random_ind] = True
    assert np.all(array[check_idx] == data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_array_xarray(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    check_idx = np.round(np.cumsum(np.random.rand(np.random.randint(1, file_n.shape[0]))).astype(int))
    np.random.shuffle(check_idx)
    assert np.all(array[check_idx] == data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_array_repeated_item_xarray(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    N = np.random.randint(2, 10)
    check_idx = np.concatenate([np.round(np.cumsum(np.random.rand(
        np.random.randint(1, file_n.shape[0]) // N)).astype(int)) for _ in range(N)])
    assert np.all(array[check_idx] == data[check_idx])


def test_simple_combination_xarray(file_map, xarray_data, xarray_files):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)
    assert np.all(array[10:20, 2:4, 3] == data[10:20, 2:4, 3])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_mask_combinations_xarray_a(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    mask = np.random.randint(0, 2, (len(stations), 5), dtype=bool)
    assert np.all(array[:, mask] == data[:, mask])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_mask_combinations_xarray_b(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    mask = np.random.randint(0, 2, (file_n.shape[0], len(stations)), dtype=bool)
    assert np.all(array[mask] == data[mask])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_expand_first_dim_xarray_a(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    check_idx = np.round(
        np.cumsum(np.random.rand((np.random.randint(4, file_n.shape[0]) // 4) * 4)).astype(int)).reshape((-1, 4))
    assert np.all(array[check_idx] == data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_expand_first_dim_xarray_b(file_map, xarray_data, xarray_files, i):
    file_n, file_idx = file_map
    data, stations = xarray_data
    data = data.to_array().values.transpose(1, 0, 2)
    array = MultifileXarray(file_n, file_idx, xarray_files, stations)

    check_idx = np.round(
        np.cumsum(np.random.rand((np.random.randint(4, file_n.shape[0]) // 4) * 4)).astype(int)).reshape((-1, 4))
    assert np.all(array[check_idx, 2:6, :] == data[check_idx, 2:6, :])
    assert np.all(array[check_idx, :, 0] == data[check_idx, :, 0])
    assert np.all(array[check_idx, 5] == data[check_idx, 5])
