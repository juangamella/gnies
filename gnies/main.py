# Copyright 2022 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import ges
from gnies.scores import GnIESScore as Score
import gnies.utils as utils

# TODO: Docstrings for known_targets
# TODO:
# - Made condition for improved score > instead of >=
# - Removed centered as argument _inner_procedure since this is defined in the score that is passed to it; add

# --------------------------------------------------------------------
# Public API


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
    # Parameters used by inner-procedure (modified GES)
    ges_iterate=True,
    ges_phases=["forward", "backward", "turning"],
    debug=0,
):
    """Runs the GnIES algorithm on the given data, producing an estimate
    of the I-equivalence class and the intervention targets.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the samples from the different environments, where
        each sample is an array with columns corresponding to
        variables and rows to observations.
    lmbda : float, default=None
        The penalization parameter for the penalized-likelihood
        score. If `None`, the BIC penalization is chosen, that is,
        `0.5 * log(N)` where `N` is the total number of observations,
        pooled across environments.
    known_targets : set, default=set()
        The set of known intervention targets, which are always kept
        as part of the target estimate of the outer procedure.
    approach : {'greedy', 'rank'}, default='greedy'
        The approach used by the outer procedure of GnIES. 'greedy'
        means that intervention targets are greedily added/removed;
        'rank' means that an ordering is found by first fitting a
        model with `I={1,...,p}`, and targets are added/removed in
        this order.
    I0 : set, default=set()
        If the 'greedy' approach is selected, specifies the initial
        set of intervention targets, to which targets are
        added/removed.
    phases : [{'forward', 'backward'}*], default=['forward', 'backward']
        If the 'greedy' approach is selected, specifies which phases
        of the outer procedure are run.
    direction : {'forward', 'backward'}, default='forward'
        If the 'rank' approach is selected, specifies whether we add or
        remove variables: if 'forward', we start with an empty
        intervention set and add targets according to the found
        ordering; if 'backward', we start with the full set and remove
        targets instead.
    ges_iterate : bool, default=False
        Indicates whether the phases of the inner procedure (modified
        GES) should be iterated more than once.
    ges_phases : [{'forward', 'backward', 'turning'}*], optional
        Which phases of the inner procedure (modified GES) are run,
        and in which order. Defaults to `['forward', 'backward',
        'turning']`.
    debug : int, default=0
        If larger than 0, debug are traces printed. Higher values
        correspond to increased verbosity.

    Returns
    -------
    score : float
        The penalized likelihood score of the GnIES estimate.
    icpdag : numpy.ndarray
        The I-CPDAG representing the estimated I-equivalence class,
        where `icpdag[i,j] != 0` implies the edge `i -> j` and
        `icpdag[i,j] != 0 & icpdag[j,i] != 0` implies the edge `i -
        j`.
    I : set of ints
        The estimate of the intervention targets.

    Raises
    ------
    ValueError:
        If an invalid value is selected for the `approach` parameter.

    """
    if approach == "greedy":
        return fit_greedy(
            data, lmbda, known_targets, I0, phases, ges_iterate, ges_phases, debug
        )
    elif approach == "rank":
        return fit_rank(
            data, lmbda, known_targets, direction, ges_iterate, ges_phases, debug
        )
    else:
        raise ValueError('Invalid value "%s" for parameter "approach"' % approach)


def fit_greedy(
    data,
    lmbda=None,
    known_targets=set(),
    I0=set(),
    phases=["forward", "backward"],
    center=True,
    ges_iterate=True,
    ges_phases=["forward", "backward", "turning"],
    debug=0,
):
    """Run the outer procedure of GnIES by greedily adding/removing
    variables until the score does not improve.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the samples from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    lmbda : float, default=None
        The penalization parameter for the penalized-likelihood
        score. If `None`, the BIC penalization is chosen, that is,
        `0.5 * log(N)` where `N` is the total number of observations,
        pooled across environments.
    known_targets : set, default=set()
        The set of known intervention targets, which are always kept
        as part of the target estimate of the outer procedure.
    I0 : set, default=set()
        Together with `known_targets`, this makes initial set of
        intervention targets, to which targets are added/removed.
    phases : [{'forward', 'backward'}*], default=['forward', 'backward']
        Specifies which phases of the outer procedure are run.
    center : bool, default=True
        Whether the data is centered when computing the score
        (`center=True`), or the noise-term means are also estimated
        respecting the constraints imposed by `I`.
    ges_phases : [{'forward', 'backward', 'turning'}*], optional
        Which phases of the inner procedure (modified GES) are run,
        and in which order. Defaults to `['forward', 'backward',
        'turning']`.
    ges_iterate : bool, default=False
        Indicates whether the phases of the inner procedure (modified
        GES) should be iterated more than once.
    debug : int, default=0
        If larger than 0, debug are traces printed. Higher values
        correspond to increased verbosity.

    Returns
    -------
    score : float
        The penalized likelihood score of the GnIES estimate.
    icpdag : numpy.ndarray
        The I-CPDAG representing the estimated I-equivalence class,
        where `icpdag[i,j] != 0` implies the edge `i -> j` and
        `icpdag[i,j] != 0 & icpdag[j,i] != 0` implies the edge `i -
        j`.
    I : set of ints
        The estimate of the intervention targets.

    """

    print("Running GnIES with greedy phases %s" % phases) if debug else None
    # Inner procedure parameters
    params = {
        "phases": ges_phases,
        "iterate": ges_iterate,
        "debug": 2 if debug > 1 else 0,
    }

    # Iteration 0: initial set + known targets
    current_I = I0 | known_targets
    score_class = Score(data, current_I, lmbda=lmbda, centered=center)
    current_estimate, current_score = _inner_procedure(score_class, current_I, **params)

    # Iterate
    p = score_class.p
    full_I = set(range(p))
    phase = "forward"
    for phase in phases:
        print("  GnIES %s phase" % phase) if debug else None
        while True:
            assert known_targets <= current_I
            print(
                "    Current I=%s (score = %0.2f)" % (current_I, current_score)
            ) if debug else None
            scores = []
            next_Is = (
                full_I - current_I if phase == "forward" else current_I - known_targets
            )
            # If no more variables can be added/removed, end this phase
            if len(next_Is) == 0:
                break
            for i in next_Is:
                new_I = current_I | {i} if phase == "forward" else current_I - {i}
                score_class.set_I(new_I)
                estimate, score = _inner_procedure(score_class, new_I, **params)
                print("      Scored I=%s : %0.2f" % (new_I, score)) if debug else None
                scores.append((score, new_I, estimate))
            # Pick the maximally scoring addition/removal
            new_score, new_I, new_estimate = max(scores)
            # If the score was improved, repeat the greedy step
            if new_score > current_score:
                current_score, current_I, current_estimate = (
                    new_score,
                    new_I,
                    new_estimate,
                )
            # Otherwise, halt
            else:
                print("    Score was not improved.") if debug else None
                break
    return current_score, current_estimate, current_I


def fit_rank(
    data,
    lmbda=None,
    known_targets=set(),
    direction="forward",
    ges_iterate=True,
    ges_phases=["forward", "backward", "turning"],
    debug=0,
):
    """Run the outer procedure of GnIES; instead of greedily
    adding/removing intervention targets, use the ordering implied by
    the variance of the noise-term-variance estimates of each
    variable. The ordering is found by first fitting a model allowing
    interventions on all targets.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the samples from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    lmbda : float, default=None
        The penalization parameter for the penalized-likelihood
        score. If `None`, the BIC penalization is chosen, that is,
        `0.5 * log(N)` where `N` is the total number of observations,
        pooled across environments.
    known_targets : set, default=set()
        The set of known intervention targets, which are always kept
        as part of the target estimate of the outer procedure.
    direction : {'forward', 'backward'}, default='forward'
        If 'forward', we start with an empty intervention set and add
        targets according to the found ordering. If 'backward', we
        start with the full set and remove targets.
    ges_phases : [{'forward', 'backward', 'turning'}*], optional
        Which phases of the inner procedure (modified GES) are run,
        and in which order. Defaults to `['forward', 'backward',
        'turning']`.
    ges_iterate : bool, default=False
        Indicates whether the phases of the inner procedure (modified
        GES) should be iterated more than once.
    debug : int, default=0
        If larger than 0, debug are traces printed. Higher values
        correspond to increased verbosity.

    Returns
    -------
    score : float
        The penalized likelihood score of the GnIES estimate.
    icpdag : numpy.ndarray
        The I-CPDAG representing the estimated I-equivalence class,
        where `icpdag[i,j] != 0` implies the edge `i -> j` and
        `icpdag[i,j] != 0 & icpdag[j,i] != 0` implies the edge `i -
        j`.
    I : set of ints
        The estimate of the intervention targets.

    """

    print("Running GnIES with %s-rank approach" % direction) if debug else None

    # Inner procedure parameters
    params = {
        "phases": ges_phases,
        "iterate": ges_iterate,
        "debug": 2 if debug > 1 else 0,
    }

    # First fit with full I to obtain an ordering based on the variance of the
    # noise-term variance estimates of each variable
    p = data[0].shape[1]
    e = len(data)
    full_I = set(range(p))
    score_class = Score(data, full_I, lmbda=lmbda)
    current_estimate, current_score = _inner_procedure(score_class, full_I, **params)
    assert utils.is_dag(current_estimate)
    _, omegas, _ = score_class._mle_full(current_estimate)
    variances = np.var(omegas, axis=0)
    order = list(np.argsort(variances))
    # Setup for the greedy outer procedure
    if direction == "forward":
        current_I = known_targets
        score_class = Score(data, current_I, lmbda=lmbda)
        current_estimate, current_score = _inner_procedure(
            score_class, current_I, **params
        )
        verb = "Adding"
        order.reverse()
    elif direction == "backward":
        current_I = full_I
        verb = "Pruning"
    else:
        raise ValueError('Invalid value "%s" for field "order"' % order)

    # Add/remove intervention targets according to the obtained
    # ordering, until the score does not improve
    print("  %s intervention targets in order %s" % (verb, order)) if debug else None
    for i in order:
        assert known_targets <= current_I
        if i in known_targets:
            continue
        print(
            "    Current I=%s (score = %0.2f)" % (current_I, current_score)
        ) if debug else None
        next_I = current_I | {i} if direction == "forward" else current_I - {i}
        score_class.set_I(next_I)
        next_estimate, next_score = _inner_procedure(score_class, next_I, **params)
        if next_score > current_score:
            current_score, current_estimate, current_I = (
                next_score,
                next_estimate,
                next_I,
            )
        else:
            break
    # Return highest scoring estimate
    return current_score, current_estimate, current_I


# --------------------------------------------------------------------
# Auxiliary functions


def _inner_procedure(
    score_class,
    I,
    lmbda=None,
    phases=["forward", "backward", "turning"],
    iterate=True,
    covariances=None,
    previous_score=None,
    debug=0,
):
    """Run the inner procedure of GnIES, i.e. GES with a modified score
    and completion algorithm.

    Parameters
    ----------
    data : list of numpy.ndarray
        A list with the samples from each environment, where each
        sample is an array with columns corresponding to variables and
        rows to observations.
    I : set of ints
        The intervention targets used for the inner procedure.
    lmbda : float, default=None
        The penalization parameter for the penalized-likelihood
        score. If `None`, the BIC penalization is chosen, that is,
        `0.5 * log(N)` where `N` is the total number of observations,
        pooled across environments.
    phases : [{'forward', 'backward', 'turning'}*], optional
        Which phases of the inner procedure (modified GES) are run,
        and in which order. Defaults to `['forward', 'backward',
        'turning']`.
    iterate : bool, default=False
        Indicates whether the phases of the inner procedure (modified
        GES) should be iterated more than once.
    covariances : numpy.ndarray, default=None
        A `e x p x p` array where `e` is the number of environmnets
        and `p` the number of variables. Specifies the scatter
        matrices used to compute the score; if `None` (default), the
        scatter matrices are the sample covariance matrices.
    debug : int, default=0
        If larger than 0, debug are traces printed. Higher values
        correspond to increased verbosity.

    Returns
    -------

    """
    # Construct completion algorithm
    assert score_class.I == I

    def completion_algorithm(PDAG):
        return utils.pdag_to_icpdag(PDAG, I)

    # Run inner procedure
    estimate, score = ges.fit(
        score_class, completion_algorithm, phases=phases, iterate=iterate, debug=debug
    )

    return estimate, score
