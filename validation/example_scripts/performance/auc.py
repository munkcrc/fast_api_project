import plotly.io as pio
from cr.plotting.plotly import ColorTheme, Theme
import cr.testing.metric as testing
import cr.calculation as calculate
from cr.data import DataSet, Segmentation
from cr.data.segmentation import ByGroup
import pandas as pd
import numpy as np
from datetime import date

# #####  Plot with plotly default template #####################################
pio.renderers.default = "browser"
theme = Theme(ColorTheme()).set_as_default()

# ##### IMPORT DATA ###########################################################
name_first = 'Jonathan'
name_last = 'Kofod'
df_in = pd.read_parquet(f"C:\\Users\\{name_first}{name_last}\\CR Consulting\\"
                        f"CR Consulting - General\\Tool\\presentation.parquet")
str_cols = df_in.select_dtypes(include=['object']).columns
df_in[str_cols] = df_in[str_cols].applymap(lambda x: np.unicode_(x))

# ### Create Dataset #################################################
dataset_original = DataSet('data', df_in)

# ### Create segmentation to remove all already defaulted #############################

segmentation_defaulted = dataset_original.segment(by='Defaulted', method=ByGroup())

dataset = segmentation_defaulted[0]

# Create a ScalarRAGResult, representing a auc value #
auc_all = testing.auc(predictions=dataset['PD 12M'], outcomes=dataset['D12'])

# The auc Result-object has some interesting outputs such as
print("auc value:", auc_all["value"].formatted_value)
print("std. dev:", auc_all["std_dev"].formatted_value)
print("color of test is", auc_all["color"].formatted_value, 'since', auc_all["reasoning"].formatted_value)
roc_curve_auc_all = auc_all["roc_curve"]
# roc_curve_auc_all.show()

# a Gini ScalarRAGResult can also be made #
gini_all = testing.gini(predictions=dataset['PD 12M'], outcomes=dataset['D12'])

# The auc Result-object has some interesting outputs such as
print("gini value:", gini_all["value"].formatted_value)
print("color of test is", gini_all["color"].formatted_value, 'since', gini_all["reasoning"].formatted_value)
cap_curve_gini_all = gini_all["cap_curve"]
# cap_curve_gini_all.show()
# notice that theoretically gini = auc*2-1
# numerical it almost true (some numerical error for calculating area under curve)
print(f'{auc_all["value"]*2-1:.5f}', f'{gini_all["value"].value:.5f}')
print('diff:', f'{auc_all["value"]*2-1-gini_all["value"]:.7f}')

# having a initial auc value, a backtest for testing if the current auc value is higher
# than then inital can be made:
auc_benchmark_all = testing.auc_benchmark(auc_initial=auc_all["value"].value,
                                          auc_current=auc_all)
print('auc_initial:', auc_benchmark_all["auc_initial"].formatted_value)
print('auc_current:', auc_benchmark_all["auc_current"]["value"].formatted_value)
print('std_dev:', auc_benchmark_all["auc_current"]["std_dev"].formatted_value)
print('h0', auc_benchmark_all["h0"])
print('p-value', auc_benchmark_all["value"].formatted_value)
print('limit_red', auc_benchmark_all["limit_red"].formatted_value)
print('limit_amber', auc_benchmark_all["limit_amber"].formatted_value)
print('test_statistic', auc_benchmark_all["test_statistic"].formatted_value)
print('color', auc_benchmark_all["color"].formatted_value)
fig_hypothesis_plot_all = auc_benchmark_all["figure_distribution"]
# fig_hypothesis_plot_all.show()


# notice that even if the initial value is slightly higher the current value, the test
# can still pass (depending on the limits of course) due to the nature of statistic
# test and uncertainty

auc_benchmark_all = testing.auc_benchmark(auc_initial=auc_all["value"]+0.009,
                                          auc_current=auc_all)
print('auc_initial:', auc_benchmark_all["auc_initial"].formatted_value)
print('auc_current:', auc_benchmark_all["auc_current"]["value"].formatted_value)
print('std_dev:', auc_benchmark_all["auc_current"]["std_dev"].formatted_value)
print('h0', auc_benchmark_all["h0"])
print('p-value', auc_benchmark_all["value"].formatted_value)
print('limit_red', auc_benchmark_all["limit_red"].formatted_value)
print('limit_amber', auc_benchmark_all["limit_amber"].formatted_value)
print('test_statistic', auc_benchmark_all["test_statistic"].formatted_value)
print('color', auc_benchmark_all["color"].formatted_value)
fig_hypothesis_plot_all = auc_benchmark_all["figure_distribution"]
# fig_hypothesis_plot_all.show()

# All of the following can be applied on sub-sets of the Dataset.

segmentation_date = dataset.segment(by='Dato', method=ByGroup())
segmentation_ep = dataset.segment(by='Erhverv/Privat', method=ByGroup())

segmentation_cross = dataset.composite_segmentations(
    *[elem for elem in dataset.segmentations])

# get the auc for Privat in '2019-06-30':
sub_set = [np.datetime64('2019-06-30'), 'Privat']
auc_2019_privat = testing.auc(
    predictions=segmentation_cross[sub_set]['PD 12M'],
    outcomes=segmentation_cross[sub_set]['D12'])
print("auc value:", auc_2019_privat["value"].formatted_value)
print("std. dev:", auc_2019_privat["std_dev"].formatted_value)
print("color of test is", auc_2019_privat["color"].formatted_value, 'since', auc_2019_privat["reasoning"].formatted_value)
roc_curve_auc_2019_privat = auc_2019_privat["roc_curve"]
# roc_curve_auc_2019_privat.show()

# and the gini
gini_2019_privat = testing.gini(
    predictions=segmentation_cross[sub_set]['PD 12M'],
    outcomes=segmentation_cross[sub_set]['D12'])
# The auc Result-object has some interesting outputs such as
print("gini value:", gini_2019_privat["value"].formatted_value)
print("color of test is", gini_2019_privat["color"].formatted_value, 'since', gini_2019_privat["reasoning"].formatted_value)
cap_curve_2019_privat = gini_2019_privat["cap_curve"]
# cap_curve_2019_privat.show()


# etc etc.


