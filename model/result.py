from pydantic import BaseModel
from output import Output


class Result(BaseModel):
    outputs: dict[Output]
    output_keys: list[str]


