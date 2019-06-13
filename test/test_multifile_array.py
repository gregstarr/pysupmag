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
    - xarray DataSets:
        - all the same

Create the test files (numpy arrays, etc.) in setup and pass in paths in a fixture
clean them up at the end with a fixture
"""


def test_a1():
    assert True
