from typing import Dict, Sequence, List, Optional
import numpy as np
import plotly.graph_objects as go


def get_buckets(bin_edges: Sequence) -> List[str]:
    def _format_value(value):
        if isinstance(value, (int, np.integer)):
            return f'{value:,.0f}'
        elif isinstance(value, float):
            max_decimals = len(f'{value}'.split('.')[1].rstrip('0'))
            if abs(value) >= 100:  # between 100 and up
                nr_of_digits = 0
            elif abs(value) >= 10:  # between 10 and 99.999
                nr_of_digits = min(1, max_decimals)
            elif abs(value) >= 1:  # between 1 and 9.999
                nr_of_digits = min(2, max_decimals)
            elif abs(value) > 0:  # between 0 and 0.999
                nr_of_digits = min(3, max_decimals)
            else:
                return '0'

            return f'{value:,.{nr_of_digits}f}'.rstrip('0').rstrip('.')
        else:
            return value
    if len(bin_edges) > 0:
        x = [f"[{_format_value(e1)} - {_format_value(e2)})"
             for e1, e2 in zip(bin_edges[:-1], bin_edges[1:])]
        x[-1] = x[-1][:-1] + "]"
    else:
        x = []
    return x


def figure_psi_deep_dive(
        psi_value: float,
        buckets: List,
        relative_frequency_a: List,
        relative_frequency_b: List,
        psi_summands: List,
        dict_non_finite: Optional[Dict] = None,
        title=None,
        name_a='a',
        name_b='b'):

    if not isinstance(psi_summands, list):
        psi_summands = list(psi_summands)
    if not isinstance(relative_frequency_a, list):
        relative_frequency_a = list(relative_frequency_a)

    if dict_non_finite is None:
        dict_non_finite = {}
    keys = dict_non_finite.keys()
    if 'relative_frequency' in keys and 'psi_summands' in keys:

        def _add_to_lists(key, bucket_name):
            if np.any([*dict_non_finite['relative_frequency'][key].values()]):
                buckets.append(bucket_name)
                psi_summands.append(dict_non_finite['psi_summands'][key])
                relative_frequency_a.append(dict_non_finite['relative_frequency'][key]['a'])
                relative_frequency_b.append(dict_non_finite['relative_frequency'][key]['b'])

        _add_to_lists('missing', 'nan')
        _add_to_lists('neg_inf', '-∞')
        _add_to_lists('pos_inf', '∞')

    def _format_percentage_psi_value(value):
        if abs(value) >= 1:  # between 1 and up
            return f'{value:.0%}'
        elif abs(value) >= 0.1:  # between 0.1 and 1
            return f'{value:.1%}'
        else:
            return f'{value:.2%}'

    def _format_percentage_psi_summands(value):
        if abs(value) >= 1:  # between 1 and up
            return f'{value:.0%}'
        elif abs(value) >= 0.1:  # between 0.1 and 1 (10% and 99.9%)
            return f'{value:.1%}'
        elif abs(value) >= 0.01:  # between 0.01 and 0.1 (1% and 9.99%)
            return f'{value:.2%}'
        else:  # less than 0.01 (1%)
            return f'{value:.3%}'

    x = [f"{elem}<br><sup>{_format_percentage_psi_summands(psi_summand)}<sup>"
         for elem, psi_summand in zip(buckets, psi_summands)]

    fig = go.Figure(data=[
        go.Bar(name=name_a, x=x, y=relative_frequency_a, text=psi_summands,
               textposition="none",
               hovertemplate='psi=%{text:.4f}<br>' + 'percent=%{y:.4f}<extra></extra>'),
        go.Bar(name=name_b, x=x, y=relative_frequency_b, text=psi_summands,
               textposition="none",
               hovertemplate='psi=%{text:.4f}<br>' + 'percent=%{y:.4f}<extra></extra>')
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')

    if len(x) < 21:
        fig.update_xaxes(tickvals=x)

    bar_gap = max(0.1, 1 / max(len(x), 2))

    psi_title = f"PSI = {_format_percentage_psi_value(psi_value)}"
    if title is None:
        title = psi_title
    else:
        title = f"{title}<br>{psi_title}"

    fig.update_layout(
        bargap=bar_gap,
        barmode='group',
        title=title,
    )

    return fig

