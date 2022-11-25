import plotly.graph_objects as go
import plotly.io as pio
from datetime import date
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Tuple


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Return (red, green, blue) for the color given as #rrggbb. """
    return tuple(int(hex_code.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(red: int, green: int, blue: int) -> str:
    """Return (red, green, blue) for the color given as #rrggbb. """
    return '#{:02x}{:02x}{:02x}'.format(red, green, blue)


def hex_to_rgba(hex_code, alpha):
    return 'rgba({},{},{},{})'.format(*hex_to_rgb(hex_code), alpha)


def rgb_to_rgba(rgb, alpha):
    return 'rgba({},{},{},{})'.format(*rgb, alpha)


class CrColorPrimary(Enum):
    BLUE_DARK = '#003749'
    BLUE_LIGHT = '#48cfe5'
    PINK = '#ff6a70'
    GREY = '#f2eeeb'
    BLACK = '#1d1d1d'
    WHITE = '#ffffff'


class CrColorSecondary(Enum):
    BEIGE_DARK = '#a18a72'
    BEIGE = '#baaa97'
    BEIGE_LIGHT = '#d6cabd'
    GREEN_DARK = '#519922'
    GREEN = '#8ddd31'
    GREEN_LIGHT = '#bcf76f'
    PURPLE = '#554084'
    PINK_DARK = '#c64692'
    PINK_LIGHT = '#cc7db2'


@dataclass
class ColorTheme:
    background: str = CrColorPrimary.GREY.value
    illustration: str = CrColorSecondary.BEIGE_LIGHT.value
    primary_text: str = CrColorPrimary.BLACK.value
    secondary_text: str = hex_to_rgba(CrColorPrimary.BLUE_DARK.value, 0.75)
    zeroline_color: str = CrColorSecondary.BEIGE.value
    pop: str = CrColorPrimary.PINK.value  # pylint: disable=unused-variable
    primary: str = CrColorPrimary.BLUE_DARK.value
    colorwheel: Tuple[str, ] = (
        CrColorPrimary.BLUE_DARK.value,
        CrColorPrimary.PINK.value,
        CrColorSecondary.PURPLE.value)


class Theme:

    def __init__(self, color_theme: ColorTheme = None):
        if color_theme is None:
            self._color_theme = ColorTheme()
        else:
            self._color_theme = color_theme

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self._color_theme, item)

    @property
    def color_theme(self) -> ColorTheme:
        return self._color_theme

    def set_as_default(self):
        key = 'default'
        add_to_pio_templates(key, self)
        pio.templates.default = key
        return self

    @contextmanager
    def modify_color(self, **kwargs):
        key = pio.templates.default
        default_template = pio.templates[key]
        color_dict = self.color_theme.__dict__
        color_dict.update(kwargs)
        add_to_pio_templates(key, Theme(ColorTheme(**color_dict)))
        try:
            yield None
        finally:
            pio.templates[key] = default_template


# https://plotly.com/python/reference/layout/

def _create_layout(theme: Theme):
    background = theme.background
    illustration = theme.illustration
    primary_text = theme.primary_text
    secondary_text = theme.secondary_text
    zeroline_color = theme.zeroline_color
    pop = theme.pop  # pylint: disable=unused-variable
    primary = theme.primary

    colorwheel = theme.colorwheel

    return dict(
        font=dict(
            color=primary_text,
            family='Barlow Medium'
        ),
        title=dict(
            xanchor='left',
            x=0.05,
            font=dict(
                family='Barlow Medium',
                size=16
            )
        ),
        paper_bgcolor=background,
        plot_bgcolor=background,
        xaxis=dict(
            gridcolor=illustration,
            showline=True,
            linecolor=secondary_text,
            tickfont=dict(color=secondary_text, size=16, family="Barlow Medium"),
            zerolinecolor=zeroline_color
        ),
        yaxis=dict(
            gridcolor=illustration,
            showline=True,
            linecolor=secondary_text,
            tickfont=dict(color=secondary_text, size=14, family="Barlow Medium"),
            zerolinecolor=zeroline_color
        ),
        margin=dict(l=60, r=45, t=55, b=50),
        colorway=colorwheel,
        legend=dict(
            title=dict(text='Legend', font=dict(size=14, color=illustration, family='Barlow Medium')),
            font=dict(size=12, color=primary_text, family='Barlow'),
            itemsizing='constant',
            orientation='h',
            valign='middle',
            x=0.5,
            xanchor='center',
            y=-0.2
        ),
        modebar=dict(
            orientation='v',
            activecolor=pop,
            color=illustration
        ),
        newshape_line_color=primary
    )


def get_theme(theme_id):
    return pio.templates[theme_id].to_plotly_json()


def add_to_pio_templates(key, theme: Theme):
    pio.templates[key] = go.layout.Template(
        layout_annotations=[
            dict(
                name='CRConsulting',
                text=f"CR: {date.today().strftime('%B %d, %Y')}",
                opacity=1,
                font=dict(color=theme.illustration,
                          size=10, family='Barlow Medium'),
                valign='middle',
                xanchor='right',
                xref='paper',
                yref='paper',
                x=1.05,
                y=1.1,
                showarrow=False,
            )
        ],
        layout=_create_layout(theme)
    )


"""
pio.templates['DEMO'] = go.layout.Template(
    layout_annotations=[
        dict(
            name='CRConsulting',
            text=f"CR: {date.today().strftime('%B %d, %Y')}", 
            opacity=1,
            font=dict(color=CrColorSecondary.BEIGE_LIGHT.value,
                      size=10, family='Barlow Medium'),
            valign='middle',
            xanchor='right',
            xref='paper',
            yref='paper',
            x=1.05,
            y=1.1,
            showarrow=False,
        )
    ],
    layout=_create_layout()
)
"""


def get_config():
    return dict(
        displaylogo=False,
        scrollZoom=False,
        responsive=False,
        modeBarButtonsToAdd=[
            'drawopenpath', 'drawcircle', 'drawline', 'eraseshape'],
        modeBarButtonsToRemove=[
            # 'toImage',
            'zoomIn2d', 'zoomOut2d', 'resetScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines', 'select2d', 'lasso2d'],
        toImageButtonOptions={"width": None, "height": None},
    )
