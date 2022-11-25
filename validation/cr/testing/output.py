from enum import Enum
from functools import total_ordering

from plotly.graph_objects import Figure
import numpy as np


class OutputType(Enum):
    SCALAR = 1
    DESCRIPTION = 2
    FIGURE = 3
    UNRESOLVED = 4
    UNKNOWN = 5
    VECTOR = 6
    MATRIX = 7


def _format_scalar(value):
    if not np.isfinite(value):
        if np.isnan(value):
            return 'nan'
        elif np.isneginf(value):
            return '-inf'
        elif np.isposinf(value):
            return 'inf'

    if abs(value) > 1e12:
        return f'{value / 1e9:,.2f}'.rstrip('0').rstrip('.') + ' B'
    if abs(value) > 1e9:
        return f'{value / 1e6:,.2f}'.rstrip('0').rstrip('.') + ' M'
    if abs(value) >= 1000:
        return f'{value:,.0f}'

    if isinstance(value, (int, np.integer)):
        return f'{value:,.0f}'
    elif isinstance(value, float):
        max_decimals = len(f'{value}'.split('.')[1].rstrip('0'))
        nr_of_decimals = 6
        # the following 6 decimals
        decimals = f'{value:.{nr_of_decimals}f}'.split('.')[1]
        # the number of 0s before the first non 0 number
        leading_decimal_zeros = len(decimals) - len(decimals.lstrip('0'))
        if leading_decimal_zeros == nr_of_decimals:
            # if all decimal are 0s
            return f'{value:,.0f}'
        elif abs(value) >= 1000:
            nr_of_digits = 0
        elif abs(value) >= 100:  # between 100 and 999.999
            nr_of_digits = min(1, max_decimals)
        elif abs(value) >= 10:  # between 10 and 99.999
            nr_of_digits = min(2, max_decimals)
        elif abs(value) >= 1:  # between 1 and 9.999
            nr_of_digits = min(3, max_decimals)
        elif abs(value) > 0:  # between 0 and 0.999
            if leading_decimal_zeros <= 3:  # between 0 and 0.000999
                nr_of_digits = min(3 + leading_decimal_zeros, max_decimals)
            elif leading_decimal_zeros <= 5:  # between 0.00001 and 0.00000999
                nr_of_digits = min(5, max_decimals)
            else:
                nr_of_digits = min(5, max_decimals)
        else:
            return '0'

        return f'{value:,.{nr_of_digits}f}'.rstrip('0').rstrip('.')
    else:
        return value


class Output(object):
    # TODO: The meaning of some output's can be dependend on other outputs, how to handle?
    def __init__(self, value, output_type=None):
        self._value = value
        if not output_type:
            output_type = self._resolve_type()
        self._output_type = output_type

    @property
    def value(self):
        if self._output_type == OutputType.UNRESOLVED:
            self._output_type = self._resolve_type(resolve=True)
            # return self._value
        # return self._value
        # TODO: remove the following lines when an operation with SourcedArray do not
        #  return a new SourcedArray but a numpy object.
        if self._output_type == OutputType.SCALAR:
            if not np.isscalar(self._value):
                return self._value[()]
            else:
                return self._value
        return self._value

    @property
    def formatted_value(self):
        if self._output_type == OutputType.UNRESOLVED:
            self._output_type = self._resolve_type(resolve=True)
        if self._output_type == OutputType.SCALAR:
            if not np.isreal(self.value):
                return self.value
            else:
                return _format_scalar(self.value)
        else:
            return self.value

    @property
    def output_type(self) -> OutputType:
        if self._output_type == OutputType.UNRESOLVED:
            self._output_type = self._resolve_type(resolve=True)
        return self._output_type

    def _resolve_type(self, resolve: bool = False):
        """
            Resolve the output type of the value.
            :param resolve: If True and the output is lazy, the value will be resolved.
        """
        if callable(self._value):
            if not resolve:
                return OutputType.UNRESOLVED
            self._value = self._value()

        if isinstance(self._value, str):
            # Unsure how to handle this? perhaps use spaces to determine?
            if len(self._value) > 16:
                return OutputType.DESCRIPTION
            return OutputType.SCALAR

        if np.isscalar(self._value):
            return OutputType.SCALAR

        if isinstance(self._value, list):
            return OutputType.DESCRIPTION

        if isinstance(self._value, Figure):
            return OutputType.FIGURE

        if isinstance(self._value, np.ndarray):
            if len(self._value.shape) == 0:
                # to capture zero-dimensional numpy arrays as SCALARS.
                # i.e. np.array(7.5) or SourcedArray(4.6)
                return OutputType.SCALAR
            if len(self._value.shape) == 1:
                return OutputType.VECTOR
            if len(self._value.shape) == 2:
                return OutputType.MATRIX

        return OutputType.UNKNOWN

    def add_source(self, recording_uid, output_key):
        self._recording_uid = recording_uid
        self._output_key = output_key

    @property
    def has_source(self):
        if hasattr(self, "_recording_uid"):
            return True
        return False

    def __repr__(self):
        return f"O({self.value.__repr__()})"

    # For ease of use, overwrite a lot of default operations trough the underlying value
    # TODO: Refine or trash

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        try:
            return getattr(self.value, attr)
        except AttributeError as err:
            raise AttributeError(f"Output({err})")

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, value):
        if isinstance(value, Output):
            return self.value.__eq__(value.value)
        return self.value.__eq__(value)

    def __ne__(self, value):
        if isinstance(value, Output):
            return self.value.__ne__(value.value)
        return self.value.__ne__(value)

    def __le__(self, value):
        if isinstance(value, Output):
            return self.value.__le__(value.value)
        return self.value.__le__(value)

    def __lt__(self, value):
        if isinstance(value, Output):
            return self.value.__lt__(value.value)
        return self.value.__lt__(value)

    def __ge__(self, value):
        if isinstance(value, Output):
            return self.value.__ge__(value.value)
        return self.value.__ge__(value)

    def __gt__(self, value):
        if isinstance(value, Output):
            return self.value.__gt__(value.value)
        return self.value.__gt__(value)

    def __iter__(self):
        return self.value.__iter__()

    def __next__(self):
        return self.value.__next__()

    def __len__(self):
        return self.value.__len__()

    def __getitem__(self, *args):
        return self.value.__getitem__(*args)

    # Addition
    def __add__(self, other):
        if isinstance(other, Output):
            return self.value + other.value
        return self.value + other

    # Reverse Addition (i.e. int + self treated as self + int)
    def __radd__(self, other):
        return self.__add__(other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Output):
            return self.value - other.value
        return self.value - other

    # Reverse Subtraction (i.e. int - self treated as -(self - int))
    def __rsub__(self, other):
        return -(self.__sub__(other))

    # Multiplication
    def __mul__(self, other):
        if isinstance(other, Output):
            return self.value * other.value
        return self.value * other

    # Reverse Multiplication (i.e. int * self treated as self * int)
    def __rmul__(self, other):
        return self.__mul__(other)

    # Division (true division, not floor division)
    def __truediv__(self, other):
        if isinstance(other, Output):
            return self.value / other.value
        return self.value / other

    # Reverse Division (i.e. int / self treated as 1/(self / int))
    def __rtruediv__(self, other):
        return 1 / (self.__truediv__(other))

    # Power of
    def __pow__(self, number):
        return self.value.__pow__(number)
