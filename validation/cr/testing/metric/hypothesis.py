from cr.automation import recordable
from cr.testing.result import ScalarRAGResult, Output
from cr.plotting.plotly import metric_plots as metric_plots
from typing import Callable, Optional


def figure_distribution(
        pdf,
        cdf_inverse,
        test_statistic,
        limits,
        colors,
        labels,
        title=None,
):
    fig = metric_plots.figure_density_curve(
        density=pdf,
        cumulative_inverse=cdf_inverse,
        name='Density'
    )
    p_ranges = [
        (elem_1, elem_2) for (elem_1, elem_2) in zip(limits[:-1], limits[1:])]
    for p_range, color, label in zip(p_ranges, colors, labels):
        metric_plots.add_color_under_curve(
            fig=fig,
            x_range=cdf_inverse(p_range),
            color=color,
            name=label
        )

    x_axis = fig.data[0]['x']
    if test_statistic < x_axis[0]:
        x = x_axis[10]
    elif x_axis[-1] < test_statistic:
        x = x_axis[-1]
    else:
        x = test_statistic

    metric_plots.add_vertical_line(
        fig=fig,
        x=x,
        name='Test Statistic'
    )
    if isinstance(title, Output):
        fig.update_layout(title=dict(text=title.value))

    return fig


def p_value_left_tailed(test_statistic, cdf):
    return cdf(test_statistic)


def p_value_right_tailed(test_statistic, cdf):
    return 1 - cdf(test_statistic)


def significance_level_left_tailed(significance_level):
    return significance_level


def significance_level_right_tailed(significance_level):
    return 1 - significance_level


def p_value_two_tailed(test_statistic, cdf):
    probability = cdf(test_statistic)
    return 2 * min(probability, 1 - probability)


def reject_h0(p_value, significance_level):
    return bool(p_value <= significance_level)


def h0_rag(
        name: str,
        test_statistic: float,
        pdf: Callable,
        cdf: Callable,
        cdf_inverse: Callable,
        calc_p_value: Callable,
        s_l_tail: Callable,
        red: float,
        amber: float,
        p_value: Optional[float] = None):

    if p_value is None:
        p_value = calc_p_value(test_statistic, cdf)

    # TODO: made pdf, cdf, cdf_inverse private attributes instead of outputs, since
    #  functions cannot be serialized at the moment.
    #  When that is fixed we can change them back to outputs.
    out = ScalarRAGResult(
        name=name, value=p_value, limit_amber=amber, limit_red=red).add_outputs(
        {"test_statistic": test_statistic,
         # "pdf": pdf,
         # "cdf": cdf,
         # "cdf_inverse": cdf_inverse
         })
    out._pdf = pdf
    out._cdf = cdf
    out._cdf_inverse = cdf_inverse

    def get_figure_distribution(output):

        limit_red = s_l_tail(out["limit_red"].value)
        limit_amber = s_l_tail(out["limit_amber"].value)

        red_hex = '#F8696B'
        amber_hex = '#FFEB84'
        green_hex = '#63BE7B'
        if limit_red < limit_amber:
            limits = (0, limit_red, limit_amber, 1)
            colors = (red_hex, amber_hex, green_hex)
            labels = ('red', 'amber', 'green')
        else:
            limits = (0, limit_amber, limit_red, 1)
            colors = (green_hex, amber_hex, red_hex)
            labels = ('green', 'amber', 'red')

        fig = figure_distribution(
            pdf=output._pdf,  # output["pdf"]._value,
            cdf_inverse=output._cdf_inverse,  # output["cdf_inverse"]._value,
            test_statistic=output["test_statistic"].value,
            limits=limits,
            colors=colors,
            labels=labels,
            title=output["figure_title"]
        )

        return fig

    return out.add_outputs({
        "figure_distribution": lambda output=out: get_figure_distribution(output)
    })


@recordable
def left_tailed_rag(
        name: str,
        test_statistic: float,
        pdf: Callable,
        cdf: Callable,
        cdf_inverse: Callable,
        p_value: Optional[float] = None,
        red: float = 0.05,
        amber: float = 0.05 * 2):
    """
    h0: mu ≥ mu_0 (calculated value ≥ to mu_0 value)
    """
    return h0_rag(
        calc_p_value=p_value_left_tailed,
        s_l_tail=significance_level_left_tailed,
        **locals()
    )


@recordable
def right_tailed_rag(
        name: str,
        test_statistic: float,
        pdf: Callable,
        cdf: Callable,
        cdf_inverse: Callable,
        p_value: Optional[float] = None,
        red: float = 0.05,
        amber: float = 0.05 * 2):
    """
    h0: mu ≤ mu_0 (calculated value ≤ to mu_0 value)
    """
    return h0_rag(
        calc_p_value=p_value_right_tailed,
        s_l_tail=significance_level_right_tailed,
        **locals()
    )
