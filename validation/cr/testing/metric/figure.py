from cr.automation import recordable
from cr.plotting.plotly import metric_plots as metric_plots, data_quality_plots as dqp
from cr.testing.result import FigureResult
from cr.data import Segmentation
from cr.data.segmentation import Temporal
import cr.testing.metric.simple as simple


@recordable
def figure_cap_curve(y_axis_model, y_axis_perfect, x_axis=None) -> FigureResult:
    return FigureResult("cap_curve",
                        metric_plots.figure_cap_curve(y_axis_model, y_axis_perfect, x_axis))


@recordable
def figure_histogram(v, name, norm='count', plot_kde=False) -> FigureResult:
    return FigureResult("Histogram",
                        dqp.figure_histogram(v, name, norm, plot_kde))


@recordable
def figure_bar_histogram(
        v, name, norm='count', unique_value_bins=True, plot_kde=False,
        unique_values=None, unique_values_count=None) -> FigureResult:
    return FigureResult(
        "Histogram",
        dqp.figure_bar_histogram(v, name, norm, unique_value_bins,
                                 plot_kde, unique_values, unique_values_count))


@recordable
def figure_histograms_as_scatter_lines(
        list_of_v, list_of_name, norm='count') -> FigureResult:
    return FigureResult(
        "Scatterplot",
        dqp.figure_histograms_as_scatter_lines(list_of_v, list_of_name, norm))


@recordable
def figure_several_histograms(list_of_v, list_of_name, norm='count') -> FigureResult:
    return FigureResult(
        "Histograms",
        dqp.figure_several_histograms(list_of_v, list_of_name, norm))


@recordable
def figure_data_quality_time_series(
        dataset, time_factor, time_resolution, metric, factor) -> FigureResult:

    segmentation_time = Segmentation(
        root_dataset=dataset, by=time_factor, method=Temporal(time_resolution.lower()))

    # TODO: need this to be more dynamic instead of this manual solution
    if metric == "COUNT":
        metric_func = simple.count
    elif metric == "MAXIMUM":
        metric_func = simple.maximum_value
    elif metric == "MEAN":
        metric_func = simple.mean_value
    elif metric == "MEDIAN":
        metric_func = simple.median
    elif metric == "MINIMUM":
        metric_func = simple.minimum_value
    elif metric == "MISSING":
        metric_func = simple.missing
    elif metric == "UNIQUE":
        metric_func = simple.unique_values
    else:
        return None

    x_axis = []
    y_axis = []
    for segment in segmentation_time.segments:
        y_axis.append(metric_func(segment[factor])["value"].value)
        x_axis.append(segment.segment_id)

    fig = dqp.figure_time_series(x_axis, y_axis)
    fig.update_layout(
        title=f"Time series plot of {metric.capitalize()} for {factor}<br><sup>",
        xaxis_title=f"{time_factor} on a {time_resolution} basis",
        yaxis_title=f"{metric.capitalize()}")

    return FigureResult("Time Series", fig)
