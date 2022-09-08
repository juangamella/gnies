# GnIES Algorithm for Causal Discovery

This repository contains a python implementation of the GnIES algorithm from the paper [*"<TODO: Title>"*](<TODO: arxiv link>) by Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml and Peter Bühlmann.

### Installation

You can clone this repo or install the python package via pip:

```bash
pip install gnies
```

There was an effort to keep dependencies on other packages to a minimum. As a result the package only depends on [`numpy`](https://numpy.org/) and [`ges`](https://github.com/juangamella/ges) (with the former being the only dependency of the latter).

## Running the algorithm

GnIES can be run through the function `gnies.fit`.

```python
def fit(
    data,
    lmbda=None,
    approach="greedy",
    # Parameters used for greedy approach
    I0=set(),
    phases=["forward", "backward"],
    # Parameters used for rank approach
    direction="forward",
    # Parameters used by inner-procedure (modified GES)
    ges_iterate=True,
    ges_phases=["forward", "backward", "turning"],
    debug=0
)
```

**Parameters**

- **data** (np.array): the matrix containing the observations of each variable (each column corresponds to a variable).
- **A0** (np.array, optional): the initial CPDAG on which GES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GES procedure are run, and in which order. Defaults to `['forward', 'backward', 'turning']`. The turning phase was found by [Hauser & Bühlmann (2012)](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) to improve estimation performace, and is implemented here too.
- **iterate** (boolean, default=False): Indicates whether the algorithm should repeat the given phases more than once, until the score is not improved.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG.
- **total_score** (float): the score of the estimate.

We offer the two approaches for selection of variables in the outer procedure of the algorithm; they can be set with the parameter `approach`, or directly through the functions [`gnies.fit_greedy`](<TODO:link>) and [`gnies.fit_rank`](<TODO:link>) in the [`gnies.main`](gnies/main.py) module. With `approach='greedy'` the greedy approach is selected, which corresponds to the results from figures 1,2 and 3 in the paper; the approach consists in greedily adding variables to the intervention targets estimate. With `approach='rank'`, the faster ranking procedure is run, at a small cost in the accuracy of the estimates (see figure <TODO: figure> in the paper).

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
_score, icpdag, I = gnies.fit_greedy(data)
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

### Example using the faster ranking approach**

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

  - `gnies.main` which is the main module with the calls to start GnIES.
  - `gnies.utils` contains auxiliary functions and the modified completion algorithm to transform PDAGs into a I-CPDAG, in the function `pdag_to_icpdag`.
  - `scores/` contains the modules with the score classes:
      - `ges.scores.decomposable_score` contains the base class for decomposable score classes (see that module for more details).
      - `ges.scores.gnies_score` contains an implementation of the cached GnIES score, as described in section 4 of the paper.
   - `test/` contains the unit tests of the scores and other components.

## Tests

All components come with unit tests to match, and some property-based tests. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There are, however, additional dependencies to run the tests. You can find these in [`requirements_tests.txt`](requirements_tests.txt).

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
