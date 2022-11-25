import numpy as np
import plotly.graph_objects as go
from scipy.stats.kde import gaussian_kde


def figure_histogram(v, name, norm='count', plot_kde=False) -> go.Figure:
    """
        norm: "percent" / "probability", "density", "probability density"
    """
    if np.issubdtype(v.dtype, np.number):
        v = v[~np.isnan(v)]
    norm = norm.lower()
    if norm in ['percent', 'probability', 'density', 'probability density']:
        norm_string = norm.capitalize()
    else:
        norm = ''
        norm_string = 'Count'
    unique_values, unique_values_count = np.unique(v, return_counts=True)
    bar_gap = max(0.1, 1 / max(len(unique_values), 2))
    fig = go.Figure(data=[go.Histogram(
        x=v, histnorm=norm,
        hovertemplate=name + '=%{x}<br>' + norm_string + '=%{y}<extra></extra>')])
    # Should we add kernel density?
    if plot_kde:
        if len(unique_values) >= 10 and norm == "probability density":
            fig.add_trace(scatter_kernel_density(v, name, norm_string, unique_values))
    if len(unique_values) < 21:
        fig.update_xaxes(tickvals=unique_values)
    fig.update_layout(
        bargap=bar_gap,
        title=f"Distribution of {name}<br><sup>"
              f"Observations in red regions are classified as outliers</sup>",
        xaxis_title=name,
        yaxis_title=norm_string)
    return fig


def unique_values_as_bins(v, unique_values=None, unique_values_count=None) -> bool:
    if np.issubdtype(v.dtype, np.number):
        v = v[~np.isnan(v)]
    else:
        return True

    out = False

    if unique_values is None and unique_values_count is None:
        unique_values, unique_values_count = np.unique(v, return_counts=True)

    if len(unique_values) < 21:
        out = True
    elif len(unique_values) < 101:

        # If all are integers
        if np.all(np.mod(v, 1) == 0):
            out = True

        # If difference between observations are less than 5 different values
        diff = unique_values[1:] - unique_values[:-1]
        unique_diff, unique_diff_count = np.unique(diff, return_counts=True)
        if len(unique_diff) < 6:
            out = True
    return out


def figure_bar_histogram(
        v, name, norm='count', unique_value_bins=True, plot_kde=False,
        unique_values=None, unique_values_count=None) -> go.Figure:
    """
        norm: "percent" / "probability", "probability density"
    """
    if np.issubdtype(v.dtype, np.number):
        v = v[~np.isnan(v)]
    if unique_value_bins:
        if unique_values is None and unique_values_count is None:
            unique_values, unique_values_count = np.unique(v, return_counts=True)
        x = unique_values
    else:
        counts, bin_edges = np.histogram(v, bins='auto')
        max_nr_of_bins = 1000
        if bin_edges.size > max_nr_of_bins + 1:
            counts, bin_edges = np.histogram(v, bins=max_nr_of_bins)
        unique_values_count = counts
        x = (bin_edges[:-1] + bin_edges[1:])/2
    norm = norm.lower()
    if norm in ['percent', 'probability', 'probability density']:
        norm_string = norm.capitalize()
        y = unique_values_count / (np.sum(unique_values_count))
    else:
        norm_string = 'Count'
        y = unique_values_count

    bar_gap = max(0.1, 1 / max(len(x), 2))

    fig = go.Figure(data=[go.Bar(
        name=name, x=x, y=y,
        hovertemplate=name + '=%{x}<br>' + norm_string + '=%{y}<extra></extra>')])

    if len(x) < 21:
        fig.update_xaxes(tickvals=x)

    if plot_kde and len(x) >= 10:
        fig.add_trace(scatter_kernel_density(v, name, norm_string))
    fig.update_layout(
        bargap=bar_gap,
        title=f"Distribution of {name}",
        xaxis_title=name,
        yaxis_title=norm_string)
    return fig


def scatter_kernel_density(v, name, value_name, x_range=None) -> go.Scatter:
    if x_range is None:
        x_range = np.linspace(np.min(v), np.max(v), len(v))
    kernel = gaussian_kde(v)
    hover_template = name + '=%{x}<br>' + value_name + '=%{y}<extra></extra>'
    return go.Scatter(x=x_range,
                      y=kernel.evaluate(x_range),
                      mode='lines',
                      line={'dash': 'solid'},
                      hovertemplate=hover_template)


def add_outlier_box_to_bar_fig(fig, lower_bound, upper_bound):
    x_axis = fig.data[0]['x']
    x_min = x_axis[0]
    x_max = x_axis[-1]
    bar_width = np.mean(x_axis[1:] - x_axis[:-1])
    sub_title = False
    if lower_bound == upper_bound:
        upper_bound = upper_bound + bar_width
        lower_bound = lower_bound - bar_width
    if x_min < lower_bound:
        fig.add_vrect(x0=x_min-(bar_width*0.5), x1=lower_bound,
                      line_width=0, fillcolor="red", opacity=0.2)
        sub_title = True
    if upper_bound < x_max:
        fig.add_vrect(x0=upper_bound, x1=x_max+(bar_width*0.5),
                      line_width=0, fillcolor="red", opacity=0.2)
        sub_title = True

    if sub_title:
        add_sub_title(fig, "Observations in red regions are classified as outliers")


def figure_histograms_as_scatter_lines(list_of_v, list_of_name, norm='count'):
    fig = go.Figure()
    for (v, name) in zip(list_of_v, list_of_name):
        fig_temp = figure_bar_histogram(
            v, name, norm=norm,
            unique_value_bins=unique_values_as_bins(v))
        x_axis = fig_temp.data[0]['x']
        y_axis = fig_temp.data[0]['y']
        fig.add_trace(go.Scatter(
            x=x_axis, y=y_axis, mode='lines', name=name,
            line={'dash': 'solid', 'width': 4}))
    return fig


def figure_several_histograms(list_of_v, list_of_name, norm='count'):
    fig = figure_bar_histogram(v=list_of_v[0], name=list_of_name[0], norm=norm)
    if len(list_of_v) > 1:
        for (v, name) in zip(list_of_v[1:], list_of_name[1:]):
            fig_temp = figure_bar_histogram(v, name, norm=norm)
            bar_obj = fig_temp.data[0]
            fig.add_trace(bar_obj)
    return fig


def figure_time_series(
        x_axis,
        y_axis
) -> go.Figure:
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis, mode='lines+markers', name='Perfect model',
        line={'dash': 'solid', 'width': 3}))
    if len(x_axis) < 21:
        fig.update_xaxes(tickvals=x_axis)
    return fig


def add_sub_title(fig, sub_title):
    current_title = fig.layout.title.text
    # <br> means new line. <sup> means subscript?
    fig.update_layout(title=dict(text=f"{current_title}<br><sup>{sub_title}</sup>"))
