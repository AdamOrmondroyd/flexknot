"""
# linf (Linear INterpolation Function)

This repo contains linfs and associated likelihoods used in my toy_sine project, and likelihoods associated with them.

If I get this right, I should be able to use this to investigate w(z) and so on.
"""

from linf.linfs import AdaptiveLinf, Linf
from linf.likelihoods import LinfLikelihood
from linf.priors import AdaptiveLinfPrior, LinfPrior
