import numpy as np
import cr.calculation as calculate
from cr.automation import recordable
from cr.testing.result import ScalarRAGResult, Result, ResultTable, ScalarResult
import cr.testing.metric.hypothesis as hypothesis
from scipy.stats import norm


@recordable
def herfindahl_index(relative_frequencies, amber=0.2, red=0.25):
    """
    The higher the value of HHI, the more concentrated the portfolio is.
    Such situation is not desirable, since a considerable part of portfolio is not
    diversified in terms of risk quality (it is being assigned the same rating grade).
    """
    h_i, coefficient_of_variation = calculate.herfindahl_index(
        relative_frequencies,
        relative_frequencies,
    )
    return ScalarRAGResult("Herfindahl Index", h_i, amber, red).add_outputs({
        "coefficient_of_variation": coefficient_of_variation
    })


@recordable
def herfindahl_index_test(cv_initial, cv_current:ScalarRAGResult, segment, red=0.05, amber=0.05 * 2):
    """
    Comparison of the Herfindahl Index at the beginning of the relevant observation
    period and the Herfindahl Index at the time of the initial validation during
    development via hypothesis testing based on a normal approximation assuming a
    deterministic Herfindahl Index at the time of the model's development. The null
    hypothesis of the test is:
    H0: current Herfindahl Index ≤  initial Herfindahl Index
    (it is upper/right-tailed (one tailed) Z-test)
    """

    if isinstance(cv_initial, Result):
        cv_initial = cv_initial["value"]
    else:
        cv_initial = cv_initial

    test_statistic, p_value = calculate.herfindahl_index_test(
        cv_initial=cv_initial,
        cv_current=cv_current["value"],
        segment=segment
    )

    right_tailed_rag = hypothesis.right_tailed_rag(
        name="Herfindahl Index Benchmark Test",
        test_statistic=test_statistic,
        pdf=norm.pdf,
        cdf=norm.cdf,
        cdf_inverse=norm.ppf,
        p_value=p_value,
        red=red,
        amber=amber).add_outputs(
            {"h0": "H0: current Herfindahl Index ≤  initial Herfindahl Index"})

    return right_tailed_rag.add_outputs({
        "cv_initial": cv_initial,
        "cv_current": cv_current,
    })

