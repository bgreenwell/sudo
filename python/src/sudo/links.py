"""Link laws for latent errors: cdf/ppf pairs used by surrogate completion.

Sign conventions (see R/sudo/surrogate.R): binary glm cloglog implies a
Gumbel-max latent error, ordinal clm cloglog implies Gumbel-min.
"""

import numpy as np
from scipy.stats import logistic


class LogisticLink:
    name = "logit"

    @staticmethod
    def cdf(x):
        return logistic.cdf(x)

    @staticmethod
    def ppf(p):
        return logistic.ppf(p)


class GumbelMaxLink:
    name = "cloglog"

    @staticmethod
    def cdf(x):
        return np.exp(-np.exp(-x))

    @staticmethod
    def ppf(p):
        return -np.log(-np.log(p))


class GumbelMinLink:
    name = "cloglog_min"

    @staticmethod
    def cdf(x):
        return 1.0 - np.exp(-np.exp(x))

    @staticmethod
    def ppf(p):
        return np.log(-np.log1p(-p))


LINKS = {l.name: l for l in (LogisticLink, GumbelMaxLink, GumbelMinLink)}


def get_link(link):
    if isinstance(link, str):
        return LINKS[link]
    return link
