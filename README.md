# GnIES Algorithm for Causal Discovery

[![PyPI version](https://badge.fury.io/py/gnies.svg)](https://badge.fury.io/py/gnies)
[![Downloads](https://static.pepy.tech/badge/gnies)](https://pepy.tech/project/gnies)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This repository contains a python implementation of the GnIES algorithm from the paper [*"Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions"*](https://arxiv.org/abs/2211.14897) by Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml and Peter Bühlmann.

If you find this code useful, please consider citing:

```
@article{gamella2022characterization,
  title={Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions},
  author={Gamella, Juan L. and Taeb, Armeen and Heinze-Deml, Christina and B\"uhlmann, Peter},
  journal={arXiv preprint arXiv:2211.14897},
  year={2022}
}
```

### Installation

You can clone this repo or install the python package via pip:

```bash
pip install gnies
```

There was an effort to keep dependencies on other packages to a minimum. As a result the package only depends on [`numpy`](https://numpy.org/) and [`ges`](https://github.com/juangamella/ges) (with the former being the only dependency of the latter).

## Running the algorithm

GnIES can be run through the function `gnies.fit`:

```python
def fit(
    data,
    lmbda=None,
    known_targets=set(),
    approach="greedy",
    # Parameters used for greedy approach
    I0=set(),
    phases=["forward", "backward"],
    # Parameters used for rank approach
    direction="forward",
    center=True,
    # Parameters used by inner-procedure (modified GES)
    ges_iterate=True,
    ges_phases=["forward", "backward", "turning"],
    debug=0
)
```

A detailed documentation can be found in the function's [docstring](https://github.com/juangamella/gnies/blob/develop/gnies/main.py#L40). The most important parameters are:

- **data** (`list of numpy.ndarray`): A list with the samples from the different environments, where each sample is an array with columns corresponding to variables and rows to observations.
- **lmbda** (`float, default=None`): The penalization parameter for the penalized-likelihood score. If `None`, the BIC penalization is chosen, that is, `0.5 * log(N)` where `N` is the total number of observations from all environments.
- **approach** (`{'greedy', 'rank'}, default='greedy'`): The approach used by the outer procedure of GnIES. With `'greedy'` targets are added and/or removed until the score does not improve; this corresponds to the results from figures 1,2 and 3 in the paper. With `'rank'`, the faster ranking procedure is run, at a small cost in the accuracy of the estimates (see figure 7 in the paper). The two procedures are implemented in `gnies.main.fit_greedy` and `gnies.main.fit_rank`, respectively.
- **center** (`bool, default=True`): Whether GnIES considers interventions on only the noise-term variance (`center=True`, i.e., data is centered before computing the score) or on both the mean and variance (`center=False`). The identifiability results in the paper correspond to (`center=True`), but (`center=False`) may improve power in some scenarios.


### Example using the greedy approach

Here [sempler](https://github.com/juangamella/sempler) is used to generate interventional data from a Gaussian SCM, but is not a dependency of the package.

```python
import sempler, sempler.generators
import gnies

# Generate a random SCM using sempler
W = sempler.generators.dag_avg_deg(10, 2.1, 0.5, 1, random_state=42)
scm = sempler.LGANM(W, (0, 0), (1, 2), random_state=42)

# Generate interventional data
n = 1000
data = [
    scm.sample(n, random_state=42),
    scm.sample(n, noise_interventions={1: (0, 11)}, random_state=42),
    scm.sample(n, noise_interventions={2: (0, 12), 3: (0, 13)}, random_state=42),
]

# Run GnIES
_score, icpdag, I = gnies.fit(data)
print(icpdag, I)

# Output:
# [[0 1 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 1 0 0 1]
#  [1 0 1 0 0 0 0 0 0 0]
#  [0 1 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 1]
#  [0 1 0 1 0 0 1 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]] {1, 2, 3}
```

### Example using the faster ranking approach

```python
# Run GnIES (on the same data as above)
_score, icpdag, I = gnies.fit(data, approach='rank')
print(icpdag, I)

# Output:
# [[0 1 0 0 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 1 0 0 1]
#  [1 0 1 0 0 0 0 0 0 0]
#  [0 1 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 1 0 0 1]
#  [0 1 0 1 0 0 1 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]] {1, 2, 3}
```

## Code Structure

The source code modules can be found inside the `gnies/` directory. These include:

  - [`gnies.main`](gnies/main.py) which is the main module with the calls to start GnIES.
  - [`gnies.utils`](gnies/utils.py) contains auxiliary functions and the modified completion algorithm to transform PDAGs into a I-CPDAG, in the function `pdag_to_icpdag`.
  - `scores/` contains the modules with the score classes:
      - [`ges.scores.decomposable_score`](gnies/scores/decomposable_score.py) contains the base class for decomposable score classes (see that module for more details).
      - [`ges.scores.gnies_score`](gnies/scores/gnies_score.py) contains an implementation of the cached GnIES score, as described in section 4 of the paper.
   - `test/` contains the unit tests of the scores and other components.

## Tests

All components come with unit tests to match, and some property-based tests. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There is, however, the additional dependency of the [`sempler`](https://github.com/juangamella/sempler) package to run the tests. You can find the details in [`requirements_tests.txt`](requirements_tests.txt).

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
