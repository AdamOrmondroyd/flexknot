"""
# Flex-Knot.

This repo contains flex-knots and associated likelihoods used in my
toy_sine project, and likelihoods associated with them.

If I get this right, I should be able to use this to investigate w(z).
"""

from flexknot.core import AdaptiveKnot, FlexKnot
from flexknot.likelihoods import Likelihood
from flexknot.priors import AdaptivePrior, Prior

__all__ = [
    "AdaptiveKnot", "FlexKnot",
    "FlexKnotLikelihood",
    "AdaptiveKnotPrior", "FlexKnotPrior",
]
