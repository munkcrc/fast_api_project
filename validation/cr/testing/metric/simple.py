import numpy as np
from typing import Optional, Union
from cr.automation import recordable
from cr.plotting.plotly import data_quality_plots as dqp
from cr.testing.result import ScalarResult, ScalarRAGResult
from cr.documentation import doc


def _rag_or_not(
        name: str,
        value: float,
        amber: Optional[float] = None,
        red: Optional[float] = None
) -> Union[ScalarResult, ScalarRAGResult]:
    if amber is not None and red is not None and not np.isnan(value):
        return ScalarRAGResult(name, value, amber, red) 
    else:
        return ScalarResult(name, value)


def _nan_handling(function, v, kwargs=None):
    if not kwargs:
        kwargs = {}
    if np.all(np.isnan(v)):
        return np.nan
    else:
        return function(v, **kwargs)


@recordable
def difference(a: float, b: float,
               amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("DIFFERENCE", a-b, amber, red).add_outputs({'a': a, 'b': b})


@recordable
def relative_difference(a: float, b: float,
                        amber: Optional[float] = None, red: Optional[float] = None):
    if b == 0:
        value = np.inf
    elif np.isnan(b):
        value = np.nan
    else:
        value = (a-b)/b
    return _rag_or_not("RELATIVE DIFFERENCE", value, amber, red).add_outputs(
        {'a': a, 'b': b})


@recordable
def ratio(a: float, b: float,
          amber: Optional[float] = None, red: Optional[float] = None):
    if b == 0:
        value = np.inf
    elif np.isnan(b):
        value = np.nan
    else:
        value = a/b
    return _rag_or_not("RATIO", value, amber, red).add_outputs({'a': a, 'b': b})


@recordable
def relative_frequency(v, amber: Optional[float] = None, red: Optional[float] = None):
    if isinstance(v, (float, int)):
        v = np.array([v])
    r_f = v / _nan_handling(np.nansum, v)
    return np.array(
        [_rag_or_not("RELATIVE FREQUENCY", elem, amber, red) for elem in r_f],
        dtype=object)


@recordable
def count(v, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("COUNT", v.size, amber, red)


@doc("""The value that appears most often""",
     output_docs={
         "count": "The number of times the mode-value appears"
     })
@recordable
def mode(v, amber: Optional[float] = None, red: Optional[float] = None):
    if isinstance(v, np.ndarray) and (
            np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.datetime64)):
        values, counts = np.unique(v[~np.isnan(v)], return_counts=True)
        index = np.argmax(counts)
        value = values[index]
        mode_count = counts[index]
    else:
        v_unicode = v.astype(np.unicode_)
        v_no_missing = v_unicode[~np.isin(v_unicode, ('nan', 'None', '', '<NA>'))]
        values, counts = np.unique(v_no_missing, return_counts=True)
        index = np.argmax(counts)
        value = values[index]
        mode_count = counts[index]
    result = _rag_or_not("MODE", value, amber, red)
    return result.add_outputs({
        "count": mode_count
    })


@recordable
def maximum_value(v, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("MAXIMUM", _nan_handling(np.nanmax, v), amber, red)


@recordable
def mean_value(v, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("MEAN", _nan_handling(np.nanmean, v), amber, red)


@recordable
def median(v, amber: Optional[float] = None, red: Optional[float] = None):
    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
        value = _nan_handling(np.nanmedian, v)
    else:
        v_unicode = v.astype(np.unicode_)
        v_no_missing = v_unicode
        if isinstance(v, np.ndarray) and not np.issubdtype(v.dtype, np.datetime64):
            v_no_missing = v_unicode[~np.isin(v_unicode, ('nan', 'None', '', '<NA>'))]
        n = len(v_no_missing)
        if n == 0:
            value = np.nan
        elif n % 2 == 1:  # if len is uneven
            value = np.sort(v_no_missing)[n // 2]
        elif n == 2:
            value = v_no_missing[0]
        else:
            value = np.sort(v_no_missing)[:-1][n // 2]
    return _rag_or_not("MEDIAN", value, amber, red)


@recordable
def minimum_value(v, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("MINIMUM", _nan_handling(np.nanmin, v), amber, red)


@doc("""Number of missing values""")
@recordable
def missing(v, amber: Optional[float] = 0.1, red: Optional[float] = 0.15):
    if isinstance(v, np.ndarray) and (
            np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.datetime64)):
        return _rag_or_not("MISSING", np.isnan(v).sum(), v.size*amber, v.size*red)
    else:
        v_unicode = v.astype(np.unicode_)
        value = np.isin(v_unicode, ('nan', 'None', '', '<NA>')).sum()
        return _rag_or_not("MISSING", value, len(v)*amber, len(v)*red)


@doc("""The q-th percentile (quantile)""",
     output_docs={"q": "The percentile score"})
@recordable
def percentile(v, q: float, amber: Optional[float] = None, red: Optional[float] = None):
    """
    q : Percentile to compute, which must be between 0 and 100 inclusive.
    """
    result = _rag_or_not(f"P{q}", _nan_handling(np.nanpercentile, v, {'q': q}), amber, red)
    return result.add_outputs({'q': q})


@recordable
def sum(v, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("SUM", _nan_handling(np.nansum, v), amber, red)


@doc("""The number of unique values""",
     output_docs={
         "mode": "The unique value that appears most often",
         "mode frequency": "The number of times the mode-value appears"
     })
@recordable
def unique_values(v, amber: Optional[float] = None, red: Optional[float] = None):
    if isinstance(v, np.ndarray) and (
            np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.datetime64)):
        values, counts = np.unique(v[~np.isnan(v)], return_counts=True)
    else:
        v_unicode = v.astype(np.unicode_)
        v_no_missing = v_unicode[~np.isin(v_unicode, ('nan', 'None', '', '<NA>'))]
        values, counts = np.unique(v_no_missing, return_counts=True)
    result = _rag_or_not("UNIQUE", len(values), amber, red)
    return result.add_outputs({
        "values": values,
        "counts": counts,
        "histogram": lambda: dqp.figure_bar_histogram(
            v=v, name='test', norm='probability',
            unique_value_bins=dqp.unique_values_as_bins(
                v, values, counts),
            unique_values=values,
            unique_values_count=counts),
        "mode": lambda vals=values, cts=counts: vals[np.argmax(cts)],
        "mode frequency": lambda cts=counts: cts[np.argmax(cts)],
    })


@doc("""The weighted average: avg = sum(factor * weights) / sum(weights)""")
@recordable
def weighted_average(v, w, amber: Optional[float] = None, red: Optional[float] = None):
    return _rag_or_not("WEIGHTED AVERAGE", _nan_handling(
        np.average, v, {'axis': None, 'weights': w, 'returned': False}), amber, red)
