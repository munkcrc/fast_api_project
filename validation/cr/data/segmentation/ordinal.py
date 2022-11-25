from typing import List, Sequence, Literal, Union
from cr.data.missing import NaNHandler
from .segmentation import SegmentationMethod
from .maps import MapByBins, MapDatesByBins
from .utilities import _repr, _str, _eq
import itertools
import numpy as np

def get_bins_with_equally_many_observations(
        x: np.ndarray,
        nr_of_bins: int = 1) -> List[float]:
    unique_x, unique_x_counts = np.unique(x, return_counts=True)
    nr_of_unique_x = unique_x.size

    possible_splits = 0.5 * unique_x[:-1] + 0.5 * unique_x[1:]

    if nr_of_unique_x == nr_of_bins:
        out = possible_splits
    elif nr_of_unique_x < nr_of_bins:
        out = possible_splits
    else:
        quantile_limits = np.linspace(0, 1, nr_of_bins + 1)
        quantiles = np.quantile(a=x, q=quantile_limits, interpolation='midpoint')
        unique_q, unique_q_counts = np.unique(quantiles, return_counts=True)
        where_q = np.where(unique_q_counts > 1)[0]
        while where_q.size:  # if a quantile occurs more than one time.
            above_1_unique_q_counts = unique_q[where_q]
            mask = ~np.isin(x, above_1_unique_q_counts)
            filtered_x = np.array(x)[mask]
            nr_of_bins_adj = (nr_of_bins-where_q.size)
            max_nr_of_obs = len(filtered_x) / nr_of_bins_adj
            adjust_x = above_1_unique_q_counts

            # find what elements in unique_x_counts that are above max_nr_of_obs
            where_above_max = np.where(unique_x_counts > max_nr_of_obs)[0]
            if where_above_max.size:
                above_max_nr_of_obs = unique_x[where_above_max]
                mask = ~np.isin(x, unique_x[where_above_max])
                filtered_x = np.array(x)[mask]
                nr_of_bins_adj = (nr_of_bins - above_max_nr_of_obs.size)
                max_nr_of_obs = len(filtered_x) / nr_of_bins_adj
                adjust_x = above_max_nr_of_obs

            # if max_nr_of_obs is not integer, then scale number of elements up in
            # filtered_x and x such that it becomes an integer. this ensures that
            # number of elem_q added is equal to optimal
            if not max_nr_of_obs.is_integer():
                decimal = (max_nr_of_obs % 1)
                if decimal <= 0.5:
                    scale = round(1 / decimal)
                else:
                    scale = 1 / decimal
                    scale = round(scale / (scale % 1))

                filtered_x = np.repeat(filtered_x, scale)
                max_nr_of_obs = len(filtered_x) // nr_of_bins_adj
                sorted_x = np.sort(np.repeat(x, scale))
            else:
                sorted_x = np.sort(x)
                max_nr_of_obs = round(max_nr_of_obs)
            # idx=where_q[0]

            for elem in adjust_x:
                sorted_x = np.concatenate([
                    sorted_x[~np.isin(sorted_x, elem)],
                    np.ones(max_nr_of_obs)*elem
                ])
            quantiles = np.quantile(
                a=sorted_x, q=quantile_limits, interpolation='midpoint')
            unique_q, unique_q_counts = np.unique(quantiles, return_counts=True)
            where_q = np.where(unique_q_counts > 1)[0]
            x = sorted_x

        mid_bins = quantiles[1:-1]
        in_possible_splits = [i for i in mid_bins if i in possible_splits]
        mask = ~np.isin(mid_bins, in_possible_splits)
        remaining_mid_bins = mid_bins[mask]
        split_domain_1 = [
            [np.max(possible_splits[possible_splits < elem]),
             np.min(possible_splits[possible_splits > elem])]
            for elem in remaining_mid_bins]
        split_domain_2 = [[t] for t in in_possible_splits]
        split_domain = split_domain_1 + split_domain_2
        # sort split_domain based in elements mean value
        split_domain.sort(key=lambda elem: sum(elem)/len(elem))
        product_iterable = itertools.product(*split_domain)
        # np.all(np.diff(t) > 0): true is all elements in t are strictly increasing
        combinations = [t for t in product_iterable if np.all(np.diff(t) > 0)]

        # t = combinations[0]
        bins = []
        measure = np.infty
        for t in combinations:
            bin_temp = np.array(t)
            bins_upper = bin_temp.copy()
            bins_upper[-1] = bins_upper[-1] + 1
            segments = np.digitize(x, bins_upper)
            unique_segments, segments_counts = np.unique(segments, return_counts=True)
            measure_temp = np.sum(np.abs(segments_counts - (len(x) / nr_of_bins)))
            if measure_temp < measure:
                bins = bin_temp
                measure = measure_temp
        out = bins
    return list(out)

def get_ordinal_bins(
        x: np.ndarray,
        nr_of_bins: int = 1,
        method: Literal['distance', 'logdistance', 'observations'] = 'distance'
) -> List[float]:
    """
    create list of bin boundaries for sequence x,
    based on nr_of_bins and method to group the bins
     """
    bins = []
    if method == "distance":
        bins = np.linspace(np.min(x), np.max(x), nr_of_bins + 1)[1:-1]
    elif method == "logdistance":
        bins = np.logspace(np.min(x), np.max(x), nr_of_bins + 1)[1:-1]
    elif method == "observations":
        bins = get_bins_with_equally_many_observations(x, nr_of_bins)
    else:
        raise ValueError(
            f"Unknown ordinal segmentation method, got {method}. "
            f"Valid includes 'distance', 'logdistance', 'observations'...")

    return list(bins)

class ByBins(SegmentationMethod):
    """
    Return list of Segment, and append the corresponding Segmentation to
    DataSet.segmentations.
    Divide values from the column self._df[by] into list of Segment, where
    the division is chosen based on nr_of_segments or bins, and approach (method)
    nan_value tells what to do with the nan in values
    (nan_value="min" sets nan_values equal to minimum value etc.)
    """

    def __init__(
            self,
            bins: Union[int, Sequence[float]] = None,
            method: Literal['distance', 'logdistance', 'observations'] = 'distance',
            nan_handling: Union[Literal['min', 'max'], float, NaNHandler] = "min"
    ):
        """
        bins : int or sequence of scalars
            If `bins` is an int, it defines the number of bins created by the method
            chosen in `method`.
            If `bins` is a sequence, it defines a monotonically increasing array of
            bin edges, including the rightmost edge, allowing for non-uniform bin
            widths. In other words, if `bins` is:

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.
        """
        if bins is None:
            raise ValueError("bins is None")

        self.bins = bins
        self.method = method
        if isinstance(nan_handling, str):
            self.nan_handler = NaNHandler(nan_handling)
        else:
            self.nan_handler = nan_handling

    def __eq__(self, other):
        return _eq(self, other, ['bins', 'method', 'nan_handler'])

    def __repr__(self):
        return _repr(self, ['bins', 'method', 'nan_handler'])

    def __str__(self):
        return _str(self, ['bins', 'method', 'nan_handler'])

    def compute_map(self, values: np.ndarray):
        values = self.nan_handler.transform(values)
        
        if isinstance(self.bins, (int, float)):
            self.bins = get_ordinal_bins(x=values,
                                         nr_of_bins=self.bins,
                                         method=self.method)
        
        # Create the map and insert the nan_handler as preprocessing
        self.map = MapByBins(self.bins)      
        self.map.pre_map = [self.nan_handler]
        return self.map

    def to_dict(self):
        return {**super().to_dict(), **{
            "bins": self.bins,
            "method": self.method,
            "nan_handling": self.nan_handler 
        }}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict['bins'], dict['method'], dict["nan_handling"])
        obj.map = MapByBins.from_dict(dict["map"])
        return obj


class TemporalByBins(SegmentationMethod):
    def __init__(
            self,
            bins: Union[int, Sequence[str]],
            time_unit: Literal['Y', 'M', 'D', 'h', 'm', 's'] = 'D',
            method: Literal['distance', 'logdistance', 'observations'] = 'distance',
            nan_handling: Union[Literal['min', 'max'], float, NaNHandler] = "min"
    ):
        self.bins = bins
        self.time_unit = time_unit
        self.method = method
        if isinstance(nan_handling, str):
            self.nan_handler = NaNHandler(nan_handling)
        else:
            self.nan_handler = nan_handling

    def __eq__(self, other):
        return _eq(self, other, ['bins', 'time_unit', 'method', 'nan_handler'])

    def __repr__(self):
        return _repr(self, ['bins', 'time_unit', 'method', 'nan_handler'])

    def __str__(self):
        return _str(self, ['bins', 'time_unit', 'method', 'nan_handler'])

    def compute_map(self, values: np.ndarray):
        values_as_int = values.astype(f'datetime64[{self.time_unit}]').astype(int)
        values_as_int = self.nan_handler.transform(values_as_int)

        if isinstance(self.bins, (int, float)):
            bins_float = get_ordinal_bins(x=values_as_int,
                                          nr_of_bins=self.bins,
                                          method=self.method)
            self.bins = [
                str(np.datetime64(round(elem), self.time_unit)) for elem in bins_float]
        else:
            self.bins = [
                str(np.datetime64(elem, self.time_unit)) for elem in self.bins]

        # Create the map and insert the nan_handler as preprocessing
        self.map = MapDatesByBins(bins=self.bins, time_unit=self.time_unit)
        self.map.pre_map = [self.nan_handler]
        return self.map

    def to_dict(self):
        return {**{
            "bins": self.bins,
            "method": self.method,
            "time_unit": self.time_unit,
            "nan_handling": self.nan_handler
        }}

    @classmethod
    def from_dict(cls, dict):
        obj = cls(dict['bins'], dict['time_unit'],  dict['method'], dict["nan_handling"])
        obj.map = MapDatesByBins.from_dict(dict["map"])
        return obj
