# Copyright 2020 Juan Luis Gamella Martin

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

"""
"""

import unittest
import numpy as np
import sempler
import sempler.generators as gen
import gnies.utils as utils
import time
import os
import gnies

NUM_GRAPHS = 2
p = 8
e = 4
n = 1000
rng = np.random.default_rng(42)


def gen_data(seed):
    W = gen.dag_avg_deg(p, 2.1, 0.5, 1, random_state=seed)
    scm = sempler.LGANM(W, (0, 0), (1, 2), random_state=seed)
    all_targets = gen.intervention_targets(p, 4, 1, replace=False, random_state=seed)
    data = [scm.sample(n)]
    for targets in all_targets:
        intervention = dict((t, (0, rng.uniform(5, 10))) for t in targets)
        sample = scm.sample(n, shift_interventions=intervention)
        data.append(sample)
    union = set.union(*[set(t) for t in all_targets])
    print("    True targets = %s" % union)
    return data


class KnownTargetsTests(unittest.TestCase):
    def test_fit(self):
        for method in ["greedy", "rank"]:
            for i in range(NUM_GRAPHS):
                print()
                print("Testing graph %d/%d" % (i + 1, NUM_GRAPHS))
                data = gen_data(i)
                targets = list(rng.permutation(range(p)))
                known_targets = set()
                while True:
                    print("  known_targets = %s" % known_targets)
                    print("    running gnies")
                    _, _, I = gnies.fit(data, approach=method, known_targets=known_targets, debug=0)
                    print("    done. Î = %s" % I)
                    self.assertTrue(known_targets <= I)
                    if len(targets) == 0:
                        break
                    else:
                        known_targets |= {targets.pop()}

    def test_greedy(self):
        options = [["forward"], ["backward"], ["forward", "backward"], ["backward", "forward"]]

        for phases in options:
            print()
            print("Phases =", phases)
            for i in range(NUM_GRAPHS):
                print()
                print("  Testing graph %d/%d" % (i + 1, NUM_GRAPHS))
                data = gen_data(i)
                targets = list(rng.permutation(range(p)))
                known_targets = set()
                while True:
                    print("    known_targets = %s" % known_targets)
                    print("      running gnies")
                    _, _, I = gnies.fit_greedy(data, phases=phases, known_targets=known_targets, debug=0)
                    print("      done. Î = %s" % I)
                    self.assertTrue(known_targets <= I)
                    if len(targets) == 0:
                        break
                    else:
                        known_targets |= {targets.pop()}

    def test_rank(self):
        directions = ["forward", "backward"]
        for direction in directions:
            print()
            print("direction =", direction)
            for i in range(NUM_GRAPHS):
                print()
                print("  Testing graph %d/%d" % (i + 1, NUM_GRAPHS))
                data = gen_data(i)
                targets = list(rng.permutation(range(p)))
                known_targets = set()
                while True:
                    print("    known_targets = %s" % known_targets)
                    print("      running gnies")
                    _, _, I = gnies.fit_rank(data, direction=direction, known_targets=known_targets, debug=0)
                    print("      done. Î = %s" % I)
                    self.assertTrue(known_targets <= I)
                    if len(targets) == 0:
                        break
                    else:
                        known_targets |= {targets.pop()}
