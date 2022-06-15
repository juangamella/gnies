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

# --------------------------------------------------------------------


def fit(data, covariances=None, I0=set(), I_candidates=None, backward_phase=False, centered=True, ges_iterate=True, ges_phases=['forward', 'backward', 'turning'], debug=0):

    print("Running alternating UT-GES") if debug else None
    # Iteration 0: no interventions
    score_class = FixedInterventionalScore(data, I0, centered=centered)
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
    I_candidates = set(range(p)) if I_candidates is None else I_candidates
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
            score_class = FixedInterventionalScore(data, new_I, centered=centered)
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
