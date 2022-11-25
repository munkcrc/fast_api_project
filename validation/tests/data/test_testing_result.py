import pytest
from cr.testing.result import *
import plotly.express as px
import numpy as np

def test_Result():
    result = Result().add_outputs({
                    "int": 1,
                    "str": "Good Result",
                    "np": np.float16("0.1"),
                    "str long": "The result was very good, and it can be seen to be so from the graph",
                    "fig": px.scatter(px.data.iris(), x="sepal_width", y="sepal_length", color="species"),
                    "callable": lambda: 1
                })
    
    assert set(result.outputs) == set(["int", "str", "str long", "np", "fig", "callable"])
    assert result["int"].output_type == OutputType.SCALAR
    assert result["str"].output_type == OutputType.SCALAR
    assert result["str"].value == "Good Result"
    assert result["np"].output_type == OutputType.SCALAR
    assert result["str long"].output_type == OutputType.DESCRIPTION
    assert result["fig"].output_type == OutputType.FIGURE
    assert result["callable"]._output_type == OutputType.UNRESOLVED
    assert result["callable"].output_type == OutputType.SCALAR
    assert result["callable"].value == 1

@pytest.mark.parametrize("scalar,red,amber,color",[
    [50, 20, 40, "GREEN"],
    [50, 40, 20, "RED"],
    [50, 25, 100, "AMBER"],
    [50, 100, 25, "AMBER"]
])
def test_ScalarRAGResult(scalar, red, amber, color):
    result = ScalarRAGResult("TEST", scalar, amber, red)

    assert result.color == color

    if result.color == "GREEN":
        assert result.is_critical == False
        assert result.is_problematic == False
        assert result.passed == True
    elif result.color == "RED":
        assert result.is_critical == True
        assert result.is_problematic == False
        assert result.passed == False
    elif result.color == "AMBER":
        assert result.is_critical == False
        assert result.is_problematic == True
        assert result.passed == False