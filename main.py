# Copyright 2021 Juan L Gamella

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

from gnies.scores.experimental import FixedInterventionalScore
import gnies.utils as utils
import ges
import numpy as np

# --------------------------------------------------------------------

def fit(data, covariances=None, centered=True, approach='rank', ges_iterate=True, ges_phases=['forward', 'backward', 'turning'], ges_lambda=None, debug=0):
    if approach == 'rank':
        return fit_rank(data, covariances, centered, ges_iterate, ges_phases, ges_lambda, debug)
    elif approach == 'greedy':
        return fit_greedy(data, covariances, set(), None, False, centered, ges_iterate, ges_phases, ges_lambda, debug)
    elif approach == 'greedy_w_backward':
        return fit_greedy(data, covariances, set(), None, True, centered, ges_iterate, ges_phases, ges_lambda, debug)
    else:
        raise ValueError('Invalid value for "approach=%s"' % approach)

def fit_rank(data, covariances=None, centered=True, ges_iterate=True, ges_phases=['forward', 'backward', 'turning'], ges_lambda=None, debug=0):
    """
    """
    
    print("Running GnIES") if debug else None

    # Fit with full I
    p = data[0].shape[1]
    e = len(data)
    full_I = set(range(p))
    score_class = FixedInterventionalScore(data, full_I, centered=centered, lmbda=ges_lambda)

    if covariances is not None:
        score_class._sample_covariances = covariances

    def completion_algorithm(PDAG):
        return utils.pdag_to_icpdag(PDAG, full_I)

    current_estimate, current_score = ges.fit(
        score_class, completion_algorithm, iterate=ges_iterate, debug=2 if debug > 1 else 0)

    # Obtain an elimination ordering based on the variance of the
    # noise-term variance estimates of each variable
    assert utils.is_dag(current_estimate)
    _, omegas = score_class._mle_full(current_estimate, [full_I] * e)
    variances = np.var(omegas, axis=0)
    order = np.argsort(variances)

    # Prune the set of intervention targets according to the obtained
    # ordering
    print("  Pruning intervention targets with order", order) if debug else None
    current_I = full_I
    for i in order:
        print("    Current I=%s (score = %0.2f)" % (current_I, current_score)) if debug else None
        next_I = current_I - {i}
        # Construct score class
        score_class = FixedInterventionalScore(data, current_I, centered=centered, lmbda=ges_lambda)
        if covariances is not None:
            score_class._sample_covariances = covariances
        # Set completion algorithm
        def completion_algorithm(PDAG):
            return utils.pdag_to_icpdag(PDAG, current_I)
        # Fit modified GES
        next_estimate, next_score = ges.fit(
            score_class, completion_algorithm, iterate=ges_iterate, debug=2 if debug > 1 else 0)
        if next_score >= current_score:
            current_score, current_estimate, current_I = next_score, next_estimate, next_I
        else:
            break
    # Return highest scoring estimate
    return current_score, current_estimate, current_I, None


        
def fit_greedy(data, covariances=None, I0=set(), I_candidates=None, backward_phase=False, centered=True, ges_iterate=True, ges_phases=['forward', 'backward', 'turning'], ges_lambda=None, debug=0):
    """
    """

    print("Running GnIES") if debug else None
    # Iteration 0: no interventions
    score_class = FixedInterventionalScore(data, I0, centered=centered, lmbda=ges_lambda)
    if covariances is not None:
        score_class._sample_covariances = covariances

    def completion_algorithm(PDAG):
        return utils.pdag_to_icpdag(PDAG, I0)
    current_estimate, current_score = ges.fit(
        score_class, completion_algorithm, iterate=ges_iterate, debug=2 if debug > 1 else 0)

    # Iterate
    p = score_class.p
    history = []
    current_I = I0
    I_candidates = set(range(p))
    phase = 'forward'
    while True:
        print("  Current I=%s (score = %0.2f)" % (current_I, current_score)) if debug else None
        history += [(current_score, current_I, current_estimate)]
        scores = []
        if phase == 'forward':
            # For each variable, compute the score resulting from adding it to I
            next_Is = I_candidates - current_I
        elif phase == 'backward':
            # For each variable in the current I estimate, compute the score resulting from removing it
            next_Is = current_I
        for i in next_Is:
            new_I = current_I | {i} if phase == 'forward' else current_I - {i}
            score_class = FixedInterventionalScore(data, new_I, centered=centered, lmbda=ges_lambda)
            if covariances is not None:
                score_class._sample_covariances = covariances

            def completion_algorithm(PDAG):
                return utils.pdag_to_icpdag(PDAG, new_I)
            estimate, score = ges.fit(score_class, completion_algorithm,
                                      iterate=ges_iterate, phases=ges_phases, debug=2 if debug > 1 else 0)
            print("    Scored I=%s : %0.2f" % (new_I, score)) if debug else None

            scores.append((score, new_I, estimate))
        # If no more variables remain, halt
        if len(scores) == 0 and (phase == 'backward' or not backward_phase):
            break
        elif len(scores) == 0:
            phase == 'backward'
            continue
        # Pick the maximally scoring i
        new_score, new_I, new_estimate = max(scores)
        # If the score was improved, repeat the greedy step
        if new_score >= current_score:
            current_score, current_I, current_estimate = new_score, new_I, new_estimate
        # Otherwise, halt
        elif phase == 'forward' and backward_phase:
            phase = 'backward'
            print("  Score was not improved. Starting backward phase.") if debug else None
        else:
            print("  Score was not improved. Halting.") if debug else None
            break
    return current_score, current_estimate, current_I, history
