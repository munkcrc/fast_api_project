import numpy as np
import plotly.graph_objects as go
from typing import Callable, Optional, Tuple
from sklearn.metrics import roc_curve
import uuid


def figure_cap_curve(
        y_axis_model,
        y_axis_perfect,
        x_axis=None
) -> go.Figure:
    if x_axis is None:
        x_axis = np.linspace(0, 1, y_axis_model.shape[0])

    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis_perfect, mode='lines', name='Perfect model',
        line={'dash': 'solid', 'width': 4}))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis_model, mode='lines', name='This model',
        line={'dash': 'solid', 'width': 4}))
    fig.add_trace(go.Scatter(
        x=x_axis, y=x_axis, mode='lines', name='Uniform model',
        line={'dash': 'dash', 'color': 'black', 'width': 4}))
    fig.update(layout_yaxis_range=[0, 1.02])
    return fig


def figure_roc_curve(
        predictions,
        outcomes
) -> go.Figure:

    # TODO: should this logic be somewhere else?
    mask_is_finite = np.isfinite(predictions) & np.isfinite(outcomes)
    predictions = predictions[mask_is_finite]
    outcomes = outcomes[mask_is_finite]

    if predictions.size == 0 or outcomes.size == 0:
        return go.Figure()

    fpr, tpr, thresholds = roc_curve(outcomes, predictions, pos_label=1)
    x_axis = np.linspace(0, 1, fpr.shape[0])

    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name='This model',
        line={'dash': 'solid', 'width': 4}))
    fig.add_trace(go.Scatter(
        x=x_axis, y=x_axis, mode='lines', name='Uniform model',
        line={'dash': 'dash', 'color': 'black', 'width': 4}))
    fig.update(layout_yaxis_range=[0, 1.02])
    return fig


def figure_density_curve(
        density: Callable,
        cumulative_inverse: Callable,
        name: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:

    if x_range is None:
        x_start = cumulative_inverse(0.0001)
        x_start = np.round(x_start, 4)
        x_end = cumulative_inverse(1 - 0.0001)
        x_end = np.round(x_end, 4)
    else:
        x_start, x_end = x_range

    x_axis_ls = np.linspace(x_start, x_end, 1000)
    step_size = 0.0001
    x_axis_a = np.arange(x_start, x_end, step_size)

    if x_axis_ls.size > x_axis_a.size:
        x_axis = x_axis_ls
    else:
        x_axis = x_axis_a

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=density(x_axis), mode='lines', name=name,
        line={'dash': 'solid', 'width': 4}))
    fig.update()
    return fig


def add_color_under_curve(
        fig,
        x_range: Tuple[float, float],
        color: str,
        name: Optional[str] = None,
) -> None:
    fig.data[0].line.color = '#1d1d1d'
    x_axis = fig.data[0]['x']
    mask = np.logical_and(x_range[0] <= x_axis, x_axis <= x_range[1])
    x_axis_filtered = x_axis[mask]
    y_axis = fig.data[0]['y']
    y_axis_filtered = y_axis[mask]

    def _interpolate_endpoint(x, start_point=True):
        """
        Having
        x_axis = [x_0, x_1, x_2, ..., x_n-2,  x_n-1, x_n] and corresponding
        y_axis = [y_0, y_1, y_2, ..., y_n-2,  y_n-1, y_n] and the subsets
        x_axis_filtered = [x_k_0, x_k_1, ..., x_k_n-1,  x_k_n] and
        y_axis_filtered = [y_k_0, y_k_1, ..., y_k_n-1,  y_k_n] for
        0 <= k_0 < k_i <  y_k_n <= n.
        Then if x lies between two x-points: x_a, x_b in x_axis where x_b = x_k_0 or
        x_a = x_k_n then we would like to add x to x_axis_filtered and
        y = y_a + (x - x_a) * (y_b - y_a)/(x_b - x_a) (linear interpolation) to
        y_axis_filtered in the start or end depending on whether x is end point in the
        beginning or end
        """
        def _find_x_a_and_x_b_and_update(x_temp, index):
            mask_left = np.max(np.where(x_axis < x_temp))
            mask_right = np.min(np.where(x_temp < x_axis))
            y_end_point = np.interp(x_temp,
                                    x_axis[mask_left:mask_right + 1],
                                    y_axis[mask_left:mask_right + 1])
            x_end_point = x_temp
            return (
                np.insert(x_axis_filtered, index, x_end_point),
                np.insert(y_axis_filtered, index, y_end_point)
            )

        interpolate = False
        if start_point:
            insert_at = 0
            if x_axis[0] < x < x_axis_filtered[0]:
                interpolate = True
        else:  # if end_point
            insert_at = len(x_axis_filtered)
            if x_axis_filtered[-1] < x < x_axis[-1]:
                interpolate = True
        if interpolate:
            return _find_x_a_and_x_b_and_update(x, insert_at)
        else:
            return x_axis_filtered, y_axis_filtered

    x_axis_filtered, y_axis_filtered = _interpolate_endpoint(x_range[0], start_point=True)
    x_axis_filtered, y_axis_filtered = _interpolate_endpoint(x_range[1], start_point=False)

    # add extra point in start and beginning so the colored area goes straight down
    x_axis_filtered = np.insert(x_axis_filtered, 0, x_axis_filtered[0])
    y_axis_filtered = np.insert(y_axis_filtered, 0, 0)

    x_axis_filtered = np.insert(x_axis_filtered, len(x_axis_filtered), x_axis_filtered[-1])
    y_axis_filtered = np.insert(y_axis_filtered, len(y_axis_filtered), 0)

    fig.add_trace(go.Scatter(
        x=x_axis_filtered, y=y_axis_filtered,
        mode='lines',
        name=name,
        # fill='tonexty',
        # fillcolor=color,
        line=dict(width=0.0, color=color),
        stackgroup=str(uuid.uuid1())  # define stack group. is unique so we don't stack
    ))
    fig.update_yaxes(range=[0, np.max(y_axis)*1.05])


def add_vertical_line(
        fig,
        x: float,
        color: str = '#1d1d1d',
        name: Optional[str] = None,
) -> None:

    x_axis = fig.data[0]['x']
    y_axis = fig.data[0]['y']
    if x_axis[0] <= x <= x_axis[-1]:
        y_max = np.max(y_axis)
        fig.add_trace(go.Scatter(
            x=[x, x], y=[0, y_max], mode='lines', name=name,
            line={'dash': 'dash', 'color': color, 'width': 3}))
