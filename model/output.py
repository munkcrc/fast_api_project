from enum import Enum
from typing import Any, Union

from pydantic import BaseModel


class Output(BaseModel):
    value: Union[float, str, int, figur]
    formatted_value: Any
    output_type: Enum
    has_source: bool
