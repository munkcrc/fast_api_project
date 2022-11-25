import cr.calculation as calculate
from cr.automation import recordable
from cr.plotting.plotly import metric_plots as metric_plots
from cr.testing.result import ScalarRAGResult, Result, ScalarResult
import cr.testing.metric.hypothesis as hypothesis
from scipy.stats import norm
import numpy as np
from cr.documentation import doc


@doc("""The area under the ROC Curve (also called AUROC). 
Used for a binary classification model, the AUC represents: 
the probability that the model will rank a randomly chosen positive example higher than 
a randomly chosen negative example. 
AUC lies between 0 and 1, and AUC = 0.5 means the model is a random predictor, 
and AUC = 1 means the prediction model is perfect.
Notice the linear relationship between Gini and AUC: Gini = 2*AUC - 1 """,
     output_docs={
         "std_dev": "The estimated standard deviation of the AUC",
         "roc_curve": "The ROC Curve which the AUC is calculated from",
     })
@recordable
def auc(predictions, outcomes, amber=0.85, red=0.7):
    # filter away nans
    mask_is_finite = np.isfinite(predictions) & np.isfinite(outcomes)
    predictions = predictions[mask_is_finite]
    outcomes = outcomes[mask_is_finite]

    if len(predictions) == 0 or len(outcomes) == 0:
        auc_value, s = (np.nan, np.nan)
    else:
        auc_value, s = calculate.auc(
            ratings=(-predictions),
            outcomes=outcomes
        )
    if auc_value is None or np.isnan(auc_value):
        return ScalarResult("AUC", np.nan)

    result_out = ScalarRAGResult("AUC", auc_value, amber, red).add_outputs({
        "std_dev": s,
        "predictions": predictions,
        "outcomes": outcomes
    })

    def get_roc_curve(output_dict):
        predictions_ = output_dict["predictions"].value
        outcomes_ = output_dict["outcomes"].value
        auc_value_ = output_dict["value"].formatted_value
        fig = metric_plots.figure_roc_curve(
            predictions=predictions_, outcomes=outcomes_)
        fig.update_layout(title=dict(text=f"ROC curve (AUC = {auc_value_})"))
        return fig
    return result_out.add_outputs({
        "roc_curve":
            lambda output_dict=result_out: get_roc_curve(output_dict)
    })


@recordable
def auc_delta(auc_initial, auc_current: ScalarRAGResult, amber=0.1, red=0.2):
    if isinstance(auc_initial, Result):
        auc_init = auc_initial["value"]
    else:
        auc_init = auc_initial
    value = auc_init - auc_current["value"]
    if value is None or np.isnan(value):
        return ScalarResult("AUC Delta", np.nan)
    return ScalarRAGResult("AUC Delta", value, amber, red).add_outputs({
        "auc_initial": auc_initial,
        "auc_current": auc_current,
    })


@doc("""A one-sided hypothesis test, testing if the AUC at the time of development is 
smaller than the AUC for the relevant observation period. The test is based on a normal
approximation, assuming a deterministic AUC at the time of development """,
     output_docs={
         "auc_initial": "AUC at the time of development",
         "auc_current": "AUC for the relevant observation period",
         "h0": "The h0 hypothesis",

     })
@recordable
def auc_benchmark(auc_initial, auc_current: ScalarRAGResult, red=0.05, amber=0.05 * 2):
    """
    H0: initial AUC ≤  current AUC (assuming a deterministic initial AUC)
    (it is lower/left-tailed Z-test BUT calculated as af right tail test)
    """
    if isinstance(auc_initial, Result):
        auc_init = auc_initial["value"]
    else:
        auc_init = auc_initial

    test_statistic, p_value = calculate.auc_benchmark_test(
        auc_initial=auc_init,
        auc_current=auc_current["value"],
        std_dev_current=auc_current["std_dev"]
    )
    if p_value is None or np.isnan(p_value):
        return ScalarResult("AUC Benchmark Test", np.nan)

    right_tailed_rag = hypothesis.right_tailed_rag(
        name="AUC Benchmark Test",
        test_statistic=test_statistic,
        pdf=norm.pdf,
        cdf=norm.cdf,
        cdf_inverse=norm.ppf,
        p_value=p_value,
        red=red,
        amber=amber).add_outputs({"h0": "H0: initial AUC ≤  current AUC"})

    return right_tailed_rag.add_outputs({
        "auc_initial": auc_initial,
        "auc_current": auc_current,
    })


@doc("""Gini (also called Accuracy Ratio (AR)) is the ratio between two areas: Gini = 𝑎𝑅/𝑎𝑃. 
𝑎𝑅 is the area between the CAP Curve of the model and the CAP Curve of the random model.
𝑎𝑃 is the area between the CAP Curve of a perfect model and the CAP Curve of the random model.
Gini lies between -1 and 1, where Gini = 0 means the model is a random predictor, 
and Gini = 1 means the prediction model is perfect.
Notice the linear relationship between Gini and AUC: AUC = (Gini + 1)/2 """,
     output_docs={
         "cap_curve": "The CAP Curve which the Gini is calculated from",
     })
@recordable
def gini(predictions, outcomes, amber=0.7, red=0.4):
    # filter away nans
    mask_is_finite = np.isfinite(predictions) & np.isfinite(outcomes)
    predictions = predictions[mask_is_finite]
    outcomes = outcomes[mask_is_finite]

    if len(predictions) == 0 or len(outcomes) == 0:
        gini_value, dict_intermediate = (np.nan, np.nan)
    else:
        gini_value, dict_intermediate = calculate.gini(
            predictions=predictions,
            outcomes=outcomes
        )
    if gini_value is None or np.isnan(gini_value):
        return ScalarResult("GINI", np.nan)

    result_out = ScalarRAGResult("GINI", gini_value, amber, red).add_outputs(
        dict_intermediate)

    def get_cap_curve(output_dict):
        x_axis = output_dict["x_axis"].value
        y_axis_model = output_dict["y_axis_model"].value
        y_axis_perfect = output_dict["y_axis_perfect"].value
        gini_value_ = output_dict["value"].formatted_value
        fig = metric_plots.figure_cap_curve(
            y_axis_model=y_axis_model,
            y_axis_perfect=y_axis_perfect,
            x_axis=x_axis)
        fig.update_layout(title=dict(text=f"CAP curve (Gini = {gini_value_})"))
        return fig
    return result_out.add_outputs({
        "roc_curve":
            lambda output_dict=result_out: get_cap_curve(output_dict)
    })


@recordable
def gini_delta(gini_initial, gini_current: ScalarRAGResult, amber=0.1, red=0.2):
    if isinstance(gini_initial, Result):
        auc_init = gini_initial["value"]
    else:
        auc_init = gini_initial
    value = auc_init - gini_current["value"]
    if value is None or np.isnan(value):
        return ScalarResult("GINI Delta", np.nan)
    return ScalarRAGResult("GINI Delta", value, amber, red).add_outputs({
        "gini_initial": gini_initial,
        "gini_current": gini_current,
    })
