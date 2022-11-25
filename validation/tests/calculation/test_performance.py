import pytest
from cr.calculation.performance import migration_matrix
import numpy as np

start_temp = [
    'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c']
end_temp = [
    'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c']
order_temp = ['b', 1, 'a', 'c', 2]

order_temp2 = np.array(order_temp, dtype=object)


@pytest.mark.parametrize("start, end, drop_nan, order, include_all, expected", [
    (start_temp, end_temp, True, order_temp, False,
     (np.array([[1.000, 0.000, 0.00],
                [1 / 3, 2 / 3, 0.00],
                [0.250, 0.000, 0.75]]),
      np.array([[5, 0, 0],
                [1, 2, 0],
                [1, 0, 3]], dtype=np.int64),
      np.array(['b', 'a', 'c'], dtype=object),
      np.array(['b', 'a', 'c'], dtype=object)
      )),
    (start_temp, end_temp, True,  np.array(order_temp, dtype=object), False, # order can also be array
     (np.array([[1.000, 0.000, 0.00],
                [1 / 3, 2 / 3, 0.00],
                [0.250, 0.000, 0.75]]),
      np.array([[5, 0, 0],
                [1, 2, 0],
                [1, 0, 3]], dtype=np.int64),
      np.array(['b', 'a', 'c'], dtype=object),
      np.array(['b', 'a', 'c'], dtype=object)
      )),
    (start_temp, end_temp, True, order_temp, True,  # include_all from order_temp
     (np.array([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [1 / 3., 0.0000, 2 / 3., 0.0000, 0.0000],
                [0.2500, 0.0000, 0.0000, 0.7500, 0.0000],
                [np.nan, np.nan, np.nan, np.nan, np.nan]]),
      np.array([[5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 2, 0, 0],
                [1, 0, 0, 3, 0],
                [0, 0, 0, 0, 0]], dtype=np.int64),
      np.array(['b', 1, 'a', 'c', 2], dtype=object),
      np.array(['b', 1, 'a', 'c', 2], dtype=object)
      )),
    (start_temp + [np.nan], end_temp + ['c'], False, order_temp, False,  # include nan
     (np.array([[1.000, 0.000, 0.00],
                [1 / 3, 2 / 3, 0.00],
                [0.250, 0.000, 0.75],
                [0.000, 0.000, 1.00]]),
      np.array([[5, 0, 0],
                [1, 2, 0],
                [1, 0, 3],
                [0, 0, 1]], dtype=np.int64),
      np.array(['b', 'a', 'c', 'nan'], dtype=object),
      np.array(['b', 'a', 'c'], dtype=object)
      )),
    (start_temp + [np.nan], end_temp + ['c'], False, order_temp, True,  # include nan and include all
     (np.array([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [1 / 3., 0.0000, 2 / 3., 0.0000, 0.0000, 0.0000],
                [0.2500, 0.0000, 0.0000, 0.7500, 0.0000, 0.0000],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]]),
      np.array([[5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 2, 0, 0, 0],
                [1, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]], dtype=np.int64),
      np.array(['b', 1, 'a', 'c', 2, 'nan'], dtype=object),
      np.array(['b', 1, 'a', 'c', 2, 'nan'], dtype=object)
      )),

])
def test_migrations_matrix(start, end, drop_nan, order, include_all, expected):
    actual = migration_matrix(start, end, drop_nan, order, include_all)
    for act, exp in zip(actual, expected):
        np.testing.assert_array_equal(np.nan_to_num(act), np.nan_to_num(exp))
