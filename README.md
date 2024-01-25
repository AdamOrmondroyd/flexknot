# Flex-Knot

This repo contains flex-knots and associated likelihoods

I initially created this for the `toy_sine` project, but it is also used for the
primordial matter power spectrum $\mathcal P_\mathcal R(k)$, and the dark energy
equation of state $w(a)$.


To pip install:

    pip install git+ssh://github.com/adamormondroyd/flexknot@main

To install locally:
    
    gh repo clone adamormondroyd/flexknot # (or ssh or https)
    cd flexknot
    pip install -e .

Or to get all test dependencies:

    pip install -e '.[dev]'

