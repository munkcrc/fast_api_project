from collections import defaultdict
import numpy as np

from .output import Output, OutputType
from typing import List, Callable, Optional, TypeVar
from pandas import DataFrame
from numpy import ndarray
ResultArray = ndarray


# TODO: Cut down on boilerplate by adding significant reflection to Results :)


class Result(object):
    """
        The Result is the most generic result container,
        it contains a dictionary of results, which both
        include nice to have and need to have results.
        These results can be computed lazily.
    """
    def __init__(self):
        self._outputs = {}

    @property
    def outputs(self):
        """ Get a list of available outputs """
        return list(self._outputs.keys())

    def __getitem__(self, item):
        """ Get a specific output """
        # If the output is a function assume that it is a 
        # lazy output, so we first compute it before returning it
        if item not in self.outputs:
            return None

        output = self._outputs[item]
        # TODO: Consider if we can attach this information another way as this is not the cleanest.
        if hasattr(self, "_recording_uid"):
            output.add_source(self._recording_uid, item)
        return self._outputs[item]

    def add_outputs(self, outputs: dict):
        """
            Add a list of outputs to the container and return it for chaining.
            :param outputs: A dictionary of outputs.
        """
        for key, value in outputs.items():
            self._outputs[key] = Output(value, output_type=None)
        return self

class MockResult(Result):
    """
        The MockResult is used to imitate a result
        Any requested outputs are themselves MockResults
    """
    def __init__(self):
        self._outputs = defaultdict(MockResult)

class FigureResult(Result):
    """
        Same as Result but added a Figure and a name attribute
    """
    def __init__(self, name, figure):
        super().__init__()        
        self.add_outputs({
            "name": name,
            "figure": figure
        })
    
    @property
    def name(self):
        return self['name']

    @property
    def figure(self):
        return self['figure']


class ScalarResult(Result):
    """
        Same as Result but added a (scalar) value and name attribute
    """
    def __init__(self, name, value):
        super().__init__()
        self.add_outputs({
            "name": name,
            "value": value
        })    
    
    @property
    def name(self):
        return self['name']

    @property
    def value(self):
        return self['value']

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'({self.name!r}, {self.value!r})')

    def __str__(self):
        return f'({self.name!r}, {self.value!r})'


class TestResult(Result):
    """
        The TestResult is the Result associated with any sort of Test, that can either
        pass or fail.
        The reasoning is a string that describes the outcome of the test.
    """
    
    def __init__(self,
                 name: str,
                 passed: bool,
                 reasoning: Optional[str] = None):
        super().__init__()
        self.add_outputs({
            "name": name,
            "passed": passed,
            "reasoning": reasoning
        })

    @property
    def name(self):
        return self["name"]

    @property
    def passed(self):
        return self["passed"]

    def __repr__(self):
        if self.passed.value:
            return f"PASSED ({self.name})"
        else:
            return f"NOT PASSED ({self.name})"

    def __str__(self):
        return f"""
            {self.__repr__()}: {self['reasoning']}
        """


class ScalarTestResult(TestResult):
    """
        Same as TestResult but added a (scalar) value attribute
    """
    def __init__(self,
                 name: str,
                 passed: bool,
                 value: float,
                 reasoning: Optional[str] = None):
        super().__init__(name, passed, reasoning)
        self.add_outputs({
            "value": value
        })

    @property
    def value(self):
        return self['value']


class RAGResult(TestResult):
    """
        The RAGResult is the TestResult associated with a test that evaluated based on
        the RAG approach. This means that the result is more granular than TestResult,
        but also that the result is linked with a opinion whether it is good,
        problematic or critical. The result implies that green is good,
        amber is problematic and red is critical.
    """
    def __init__(self,
                 name: str,
                 color: str,
                 reasoning: Optional[str] = None):
        assert color in ["RED", "AMBER", "GREEN"]
        is_good = color == "GREEN"
        is_critical = color == "RED"
        is_problematic = color == "AMBER"
        super().__init__(name, passed=not is_critical, reasoning=reasoning)
        self.add_outputs({
            "color": color,
            "is_good": is_good,
            "is_critical": is_critical,
            "is_problematic": is_problematic,
        })

    @property
    def color(self):
        return self['color']

    @property
    def is_good(self):
        return self["is_good"]

    @property
    def is_critical(self):
        return self["is_critical"]

    @property
    def is_problematic(self):
        return self["is_problematic"]

    def __repr__(self):
        if self.is_critical.value:
            return f"CRITICAL ({self.name})"
        if self.is_problematic.value:
            return f"PROBLEMATIC ({self.name})"
        return f"GOOD ({self.name})"


class ScalarRAGResult(RAGResult):
    """
        The ScalarRAGResult is the RAGResult based on a scalar value.
    """
    def __init__(self, name: str, value: float, limit_amber: float, limit_red: float):
        color, reason = self.scalar_to_color(value, limit_amber, limit_red)
        super().__init__(name=name, color=color, reasoning=reason)
        self.add_outputs({
            "value": value,
            "limit_red": limit_red,
            "limit_amber": limit_amber
        })
        
    @property
    def value(self):
        return self['value']

    @property
    def limit_red(self):
        return self['limit_red']
        
    @property
    def limit_amber(self):
        return self['limit_amber']

    def scalar_to_color(self, value, limit_amber, limit_red):
        # Determine if higher is better then evaluate color
        if limit_red < limit_amber:
            if value < limit_red:
                return "RED", f"Value ({value:.4f}) below red threshold ({limit_red})"
            elif value < limit_amber:
                return "AMBER", f"Value ({value:.4f}) below amber threshold ({limit_amber})"
            return "GREEN", f"Value ({value:.4f}) above amber threshold ({limit_amber})"
        else:
            if value > limit_red:
                return "RED", f"Value ({value:.4f}) above red threshold ({limit_red})"
            elif value > limit_amber:
                return "AMBER", f"Value ({value:.4f}) above amber threshold ({limit_amber})"
            return "GREEN", f"Value ({value:.4f}) below amber threshold ({limit_amber})"


class ResultTable(Result):
    """
        The ResultTable is a table structure of Results-type objects:

                        column_names[0], column_names[1], ..., column_names[k]
        row_names[0]      results[0, 0],   results[0, 1], ...,   results[0, k]
        row_names[1]      results[1, 0],   results[1, 1], ...,   results[1, k]
        ...
        row_names[n]      results[n, 0],   results[n, 1], ...,   results[n, k]

        where each entry in row_names and column_names is a string, and each entry in
        results is a Result object

    """

    def __init__(self, name: str, row_names: List[str], column_names: List[str], results: ResultArray):
        super().__init__()
        self.add_outputs({
            "name": name,
            "row_names": row_names,
            "column_names": column_names,
            "results": results
        })
        
    @property
    def name(self):
        return self['name']
        
    @property
    def row_names(self):
        return self['row_names']

    @property
    def column_names(self):
        return self['column_names']
        
    @property
    def results(self):
        return self['results']

    @results.setter
    def results(self, result_updated):
        self.add_outputs({
            "results": result_updated
        })

    def get_result_subset(self, row_names=None, column_names=None) -> ResultArray:
        indices_row = []
        indices_column = []
        if row_names is not None:
            if isinstance(row_names, (str, float, int)):
                row_names = [row_names]
            indices_row = [self.row_names.index(name) for name in row_names]
        if column_names is not None:
            if isinstance(column_names, (str, float, int)):
                column_names = [column_names]
            indices_column = [self.column_names.index(name) for name in column_names]
        if indices_row and indices_column:
            return self.results[np.ix_(indices_row, indices_column)]
        elif indices_row:
            return self.results[indices_row, :]
        else:  # indices_column:
            return self.results[:, indices_column]

    def get_column_results(self, column_names, attribute="value", value=False):
        """
        If columns is a string or list of len 1 return List[Result]
        Else return List[List[Result]] with "shape" (# self.row_names, # cols)
        """
        sub_results = self.get_result_subset(column_names=column_names)
        if value:
            return np.array(
                [[entry[attribute].value for entry in out_row] for out_row in sub_results])
        else:
            return np.array(
                [[entry[attribute] for entry in out_row] for out_row in sub_results],
                dtype=self.results.dtype)

    def get_row_results(self, row_names, attribute="value", value=False):
        """
        If rows is a string or list of len 1 return List[Result]
        Else return List[List[Result]] with "shape" (# rows, # self.row_names)
        """
        sub_results = self.get_result_subset(row_names=row_names)
        if value:
            return np.array(
                [[entry[attribute].value for entry in out_row] for out_row in sub_results])
        else:
            return np.array(
                [[entry[attribute] for entry in out_row] for out_row in sub_results],
                dtype=self.results.dtype)

    def insert_column(self, column_values, column_name, idx):
        self.results = np.c_[
            self.results.value[:, :idx],
            np.array([[ScalarResult(column_name, value)] for value in np.squeeze(column_values, axis=1)]),
            self.results.value[:, idx:]
        ]
        self.column_names.insert(idx, column_name)

    def to_dataframe(self, attribute="value", value=False) -> DataFrame:
        attribute_data = [[result[attribute] for result in results]
                          for results in self.results.value]
        df = DataFrame(attribute_data,
                       columns=self.column_names.value,
                       index=self.row_names.value)
        if value:
            df = df.applymap(lambda x: x.value if isinstance(x, Output) else x)

        return df
