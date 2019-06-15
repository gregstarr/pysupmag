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
from tempfile import TemporaryDirectory
from pysupmag.multifile_array import MultifileNumpyArray

N_TESTS = 10000


@pytest.fixture(scope="module")
def file_map():
    lengths = np.random.randint(1, 200, 3)
    file_idx = np.concatenate([np.arange(lengths[i]) for i in range(3)]).astype(int)
    file_n = np.concatenate([np.ones(lengths[i]) * i for i in range(3)]).astype(int)
    return file_n, file_idx


@pytest.fixture(scope="module")
def numpy_data(file_map):
    file_n, file_idx = file_map
    data = np.random.rand(file_n.shape[0], 10, 10)
    return data


@pytest.fixture(scope="module")
def numpy_files(file_map, numpy_data):
    file_n, file_idx = file_map
    with TemporaryDirectory() as tempdirname:
        paths = []
        for i in range(3):
            fn = "{}\\{}.npy".format(tempdirname, i)
            np.save(fn, numpy_data[file_n == i])
            paths.append(fn)
        yield paths


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_numpy(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    check_idx = np.random.randint(file_n.shape[0])
    assert np.all(array[check_idx] == numpy_data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_slices_numpy(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))

    check_idx = slice(np.random.randint(1, file_n.shape[0]))
    assert np.all(array[check_idx] == numpy_data[check_idx])

    a = np.random.randint(1, file_n.shape[0] - 1)
    b = np.random.randint(1, file_n.shape[0] - a)
    check_idx = slice(a, a + b)
    assert np.all(array[check_idx] == numpy_data[check_idx])

    a = np.random.randint(10, file_n.shape[0] - 10)
    b = np.random.randint(5, file_n.shape[0] - a - 5)
    c = np.random.randint(1, b - 1)
    check_idx = slice(a, a + b, c)
    assert np.all(array[check_idx] == numpy_data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_boolean_mask_numpy(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    random_ind = np.random.randint(0, file_n.shape[0], np.random.randint(1, file_n.shape[0]))
    check_idx = np.zeros_like(file_n, dtype=bool)
    check_idx[random_ind] = True
    assert np.all(array[check_idx] == numpy_data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_array_numpy(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    check_idx = np.round(np.cumsum(np.random.rand(np.random.randint(1, file_n.shape[0]))).astype(int))
    assert np.all(array[check_idx] == numpy_data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_integer_array_repeated_item_numpy(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    N = np.random.randint(2, 10)
    check_idx = np.concatenate([np.round(np.cumsum(np.random.rand(
        np.random.randint(1, file_n.shape[0]) // N)).astype(int)) for _ in range(N)])
    assert np.all(array[check_idx] == numpy_data[check_idx])


def test_simple_combination_numpy(file_map, numpy_data, numpy_files):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    assert np.all(array[10:20, 2:4, 8] == numpy_data[10:20, 2:4, 8])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_mask_combinations_numpy_a(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    mask = np.random.randint(0, 2, (10, 10), dtype=bool)
    assert np.all(array[:, mask] == numpy_data[:, mask])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_mask_combinations_numpy_b(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    mask = np.random.randint(0, 2, (file_n.shape[0], 10), dtype=bool)
    assert np.all(array[mask] == numpy_data[mask])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_expand_first_dim_numpy_a(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    check_idx = np.round(
        np.cumsum(np.random.rand((np.random.randint(4, file_n.shape[0]) // 4) * 4)).astype(int)).reshape((-1, 4))
    assert np.all(array[check_idx] == numpy_data[check_idx])


@pytest.mark.parametrize("i", [0] * N_TESTS)
def test_expand_first_dim_numpy_b(file_map, numpy_data, numpy_files, i):
    file_n, file_idx = file_map
    array = MultifileNumpyArray(file_n, file_idx, numpy_files, (10, 10))
    check_idx = np.round(
        np.cumsum(np.random.rand((np.random.randint(4, file_n.shape[0]) // 4) * 4)).astype(int)).reshape((-1, 4))
    assert np.all(array[check_idx, 2:6, :] == numpy_data[check_idx, 2:6, :])
    assert np.all(array[check_idx, :, 0] == numpy_data[check_idx, :, 0])
    assert np.all(array[check_idx, 5] == numpy_data[check_idx, 5])
