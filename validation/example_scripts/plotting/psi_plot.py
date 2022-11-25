import numpy as np
import cr.plotting.plotly as cr_plot
import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
# #####  Plot with plotly default template #####################################
pio.renderers.default = "browser"
theme = Theme(ColorTheme()).set_as_default()

# ## numerical psi
dict_intermediate = {
    'bin_edges': np.array([21., 32.6, 44.2, 55.8, 67.4, 79.]),
    'relative_frequency': {
        'a': np.array([0.29346259, 0.29728582, 0.19302717, 0.14303703, 0.07245915]),
        'b': np.array([0.2961321, 0.2941089, 0.1880839, 0.14089557, 0.08015472])},
    'psi_summands': np.array(
        [2.41736452e-05, 3.41325937e-05, 1.28242270e-04, 3.23032107e-05,
         7.76757647e-04]),
    'non_finite': {'relative_frequency': {'missing': {'a': 0.0007282326703381731,
                                                      'b': 0.0006248140434394526},
                                          'neg_inf': {'a': 0.0, 'b': 0.0},
                                          'pos_inf': {'a': 0.0, 'b': 0.0}},
                   'psi_summands': {'missing': 1.5840271627798136e-05,
                                    'neg_inf': 0,
                                    'pos_inf': 0}}}

psi_value = 0.001011449638532344
bin_edges = dict_intermediate["bin_edges"]
relative_frequency_a = list(dict_intermediate['relative_frequency']['a'])
relative_frequency_b = list(dict_intermediate['relative_frequency']['b'])
psi_summands = list(dict_intermediate["psi_summands"])
dict_non_finite = dict_intermediate['non_finite']

fig = cr_plot.figure_psi_deep_dive(psi_value, cr_plot.get_buckets(bin_edges), relative_frequency_a, relative_frequency_b, psi_summands, dict_non_finite, name_a='subset a', name_b='subset b')
fig.show()


# ## categorical psi
dict_intermediate = {
    'buckets': np.array([0, 1, 2, 3, 4], dtype=np.int64),
    'relative_frequency': {
        'a': np.array([0.12962542, 0.23980095, 0.00439974, 0.58433086, 0.04184304]),
        'b': np.array([0.12787861, 0.2380244, 0.00380839, 0.58908063, 0.04120797])},
    'psi_summands': np.array(
        [2.36997009e-05, 1.32104854e-05, 8.53544480e-05, 3.84527374e-05,
         9.71237378e-06])}

psi_value = 0.00017042974554486097
buckets = list(dict_intermediate['buckets'])
relative_frequency_a = list(dict_intermediate['relative_frequency']['a'])
relative_frequency_b = list(dict_intermediate['relative_frequency']['b'])
psi_summands = list(dict_intermediate["psi_summands"])

fig = cr_plot.figure_psi_deep_dive(psi_value, buckets, relative_frequency_a, relative_frequency_b, psi_summands, name_a='subset a', name_b='subset b')
fig.show()


