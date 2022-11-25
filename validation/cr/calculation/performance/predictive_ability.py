import numpy as np
from scipy.stats import beta


def jeffreys_test(
        n,
        x,
        applied_p,
        significance_level: float = 0.05,
        one_sided=True
):
    """
    Suppose X = x ∼ Bin(n,p) and suppose p has a prior distribution Beta(α_1, α_2).
    Then the posterior distribution of p is Beta(x + α_1, n - x + α_2)
    For Jeffreys test α_1, α_2 is set to be equal to 0.5 and is called Jeffreys priors.
    The endpoints of the Jeffreys prior interval are the α/2 and 1−α/2 quantiles
    (α=significance_level) of the Beta(x + 0.5, n - x + 0.5) distribution
    H0: P applied ≥ True P (it is lower/left-tailed test)

    """

    a = x + 0.5
    b = n - x + 0.5

    p_value = beta.cdf(applied_p, a, b, loc=0, scale=1)

    if one_sided:
        prior_interval = [
            beta.ppf(significance_level, a, b, loc=0, scale=1),
            np.inf]
        hypotheses = "H0: PD applied ≥ True PD"
        conclusion = one_sided_test(p_value, significance_level)
    else:
        prior_interval = [
            beta.ppf(significance_level/2, a, b, loc=0, scale=1),
            beta.ppf(1-significance_level/2, a, b, loc=0, scale=1)]
        hypotheses = "H0: PD applied = True PD"
        conclusion = two_sided_test(p_value, significance_level)

    return p_value, prior_interval, hypotheses, conclusion


def one_sided_test(
        p_value,
        significance_level
):
    str_sl = "{:.2%}".format(significance_level)
    str_p_val = "{:.2%}".format(p_value)
    if str_sl == str_p_val:
        str_p_val = "{:.3%}".format(p_value)

    if p_value <= significance_level:
        string_out = \
            f"There is evidence at the {str_sl} level to reject H0 " \
            f"since p-value = {str_p_val} ≤ {str_sl} = α"
    else:
        string_out = \
            f"There is no evidence at the {str_sl} level to reject H0 " \
            f"since α = {str_sl} < {str_p_val} = p-value"
    return string_out


def two_sided_test(
        p_value,
        significance_level
):
    str_sl = "{:.2%}".format(significance_level)
    str_sl_l = "{:.2%}".format(significance_level / 2)
    str_sl_r = "{:.2%}".format(1 - significance_level / 2)
    str_p_val = "{:.2%}".format(p_value)
    if str_sl_l == str_p_val or str_sl_r == str_p_val:
        str_p_val = "{:.3%}".format(p_value)

    if not (significance_level / 2 < p_value < 1 - significance_level / 2):
        string_out = \
            f"There is no evidence at the {str_sl} level to reject H0 " \
            f"since α/2 = {str_sl_l} < {str_p_val} (p-value) < {str_sl_r} = 1-α/2"
    elif p_value <= significance_level / 2:
        string_out = \
            f"There is evidence at the {str_sl} level to reject H0 " \
            f"since p-value = {str_p_val} ≤ {str_sl_l} = α/2"
    else:
        string_out = \
            f"There is evidence at the {str_sl} level to reject H0 " \
            f"since 1-α/2 = {str_sl_r} ≤ {str_p_val} = p-value"
    return string_out