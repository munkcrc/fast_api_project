import numpy as np
import cr.calculation as calculate
from cr.automation import recordable
import cr.testing.metric.hypothesis as hypothesis
from scipy.stats import beta
from functools import partial
from cr.testing.result import ScalarResult
from cr.documentation import doc


@doc("""A one-sided hypothesis test, testing if the probability applied in a sample is 
greater than the true probability in the same sample. The test is based on the 
assumption that the number of positive outcomes in the sample follows X = x ∼ Bin(n,p) 
where p follows the posterior distribution of Beta(x + 0.5, n - x + 0.5)""",
     output_docs={
         "n": "Number of observations",
         "x": "Number of observed true outcomes",
         "applied_p": "The applied probability",
         "h0": "The h0 hypothesis",
     })
@recordable
def jeffreys_test(n, x, applied_p, red=0.05, amber=0.05 * 2):
    """
    Suppose X = x ∼ Bin(n,p) and suppose p has a prior distribution Beta(α_1, α_2).
    Then the posterior distribution of p is Beta(x + α_1, n - x + α_2)
    For Jeffreys test α_1, α_2 is set to be equal to 0.5 and is called Jeffreys priors.
    The endpoints of the Jeffreys prior interval are the α/2 and 1−α/2 quantiles
    (α=significance_level) of the Beta(x + 0.5, n - x + 0.5) distribution
    H0: p applied ≥ true p (it is lower/left-tailed test)
    """

    if np.isnan(n) or np.isnan(x) or np.isnan(applied_p):
        return ScalarResult(
                name="Jeffreys test",
                value=np.nan)

    a = x + 0.5
    b = n - x + 0.5

    pdf = partial(beta.pdf, a=a, b=b, loc=0, scale=1)
    cdf = partial(beta.cdf, a=a, b=b, loc=0, scale=1)
    cdf_inverse = partial(beta.ppf, a=a, b=b, loc=0, scale=1)

    prior_interval = [cdf_inverse(red), 1]

    left_tailed_rag = hypothesis.left_tailed_rag(
        name="Jeffreys test",
        test_statistic=applied_p,
        pdf=pdf,
        cdf=cdf,
        cdf_inverse=cdf_inverse,
        red=red,
        amber=amber).add_outputs({"h0": "p applied ≥ true p"})

    return left_tailed_rag.add_outputs({
        "n": n,
        "x": x,
        "applied_p": applied_p,
        "true probability": x/n,
        "prior_interval": prior_interval,
    })
