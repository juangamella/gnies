# GnIES Algorithm for Causal Discovery

This is a python implementation of the GnIES algorithm from the paper [*"<TODO: Title>"*](<TODO: arxiv link>) by Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml and Peter Bühlmann.

## Installation

You can clone this repo or install the python package via pip:

```bash
pip install gnies
```

## Running the algorithm

### Using the greedy approach for the outer procedure

Lorem Ipsum.

```python
ges.fit_bic(data, A0 = None, phases = ['forward', 'backward', 'turning'], debug = 0)
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

**Example**

Here [sempler](https://github.com/juangamella/sempler) is used to generate an observational sample from a Gaussian SCM, but this is not a dependency.

```python
import ges
import sempler
import numpy as np

# Generate observational data from a Gaussian SCM using sempler
A = np.array([[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0]])
W = A * np.random.uniform(1, 2, A.shape) # sample weights
data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)

# Run GES with the Gaussian BIC score
estimate, score = ges.fit_bic(data)

print(estimate, score)

# Output
# [[0 0 1 0 0]
#  [0 0 1 0 0]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 1 0]] 21511.315220683457
```

### Using the faster ranking approach


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
