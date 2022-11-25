from .templating import get_config, get_theme, ColorTheme, Theme
# import plotly.graph_objects as go
# import plotly.io as pio
from .metric_plots import *
from .data_quality_plots import *
from .psi_plot import *

def export_figure(fig, path, format="svg"):
    fig.write_image(path, engine="orca", format=format)  # engine="kaleido"
