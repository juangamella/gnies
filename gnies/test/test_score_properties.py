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
import sempler.generators
import gnies.utils as utils
# import causalicp as icp
import pickle
import time

# from termcolor import colored
from gnies.scores import GnIESScore as Score

NUM_GRAPHS = 100

# ---------------------------------------------------------------------
# Tests


class Assumption1Tests(unittest.TestCase):
    """Tests for assumption 1: Score equivalence"""
    seed = 23
    np.random.seed(seed)
    p = 20
    k = 2.7
    e = 5
    int_size = (1, 4)
    # SCM weights, means and variances
    W = (1, 2)
    means = (-1, 1)
    variances = (1, 2)
    # Intervention means/variances
    i_means = (-2, 2)
    i_vars = (4, 5)
    n = np.random.randint(100, 1000)

    # Build SCM
    W = sempler.generators.dag_avg_deg(p, k, W[0], W[1])
    true_A = (W != 0).astype(int)
    scm = sempler.LGANM(W, means, variances)
    true_means = scm.means
    true_variances = scm.variances

    # Build intervention targets
    true_targets = sempler.generators.intervention_targets(
        p, e - 1, int_size)
    true_targets = [set()] + [set(I) for I in true_targets]
    true_targets_union = set.union(*true_targets)
    interventions = []
    for targets in true_targets:
        means = np.random.uniform(i_means[0], i_means[1], len(targets))
        variances = np.random.uniform(i_vars[0], i_vars[1], len(targets))
        interventions += [dict(zip(targets, zip(means, variances)))]

    # Sample data
    data = []
    for I in interventions:
        data.append(scm.sample(n, shift_interventions=I, random_state=42))

    # Set up score instance
    gnies_score = Score(data, true_targets_union, centered=False)

    data_generating_vars = locals()
    # Debug outputs
    print(data_generating_vars)

    # ------------------------------------------------------
    # Tests for assumption 1

    def test_score_equivalence_1(self):
        # Test that I-equivalent DAGs entail the same score
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, self.k, random_state=i)
                   for i in range(G)]
        # Test
        for i, A in enumerate(graphs):
            I = set(sempler.generators.intervention_targets(
                self.p, 1, (0, self.p))[0])
            i_cpdag = utils.dag_to_icpdag(A, I)
            equivalent_dags = utils.all_dags(i_cpdag)
            gnies_score = Score(self.data, I)
            # Iterate over all equivalent DAGs, and check they have the same score
            first_score = gnies_score.full_score(A)
            print("\nClass %d/%d, I = %s" % (i + 1, G, I))
            for j, D in enumerate(equivalent_dags):
                score = gnies_score.full_score(D)
                dif_edges = (A != D).sum() / 2
                print("  I-ME %d/%d (%d dif. edges) :\t%0.32f" %
                      (j + 1, len(equivalent_dags), dif_edges, score))
                self.assertTrue(np.isclose(first_score, score))
        print("Tested score equivalence for %d equivalence classes" % (i + 1))


# class Assumption4Tests(unittest.TestCase):
#     """Tests for assumption 4: Local consistency (invariance)"""

#     seed = 41
#     np.random.seed(seed)
#     p = 7
#     k = 2.7
#     e = 3
#     max_int_size = 2
#     # SCM weights, means and variances
#     W = (4, 5)
#     means = (0, 0)
#     variances = (1, 2)
#     # Intervention means/variances
#     i_means = (0, 0)
#     i_vars = (4, 5)  # 10, 15)
#     n = round(1e7)  # np.random.randint(100, 10000)

#     # Build SCM
#     W = sempler.generators.dag_avg_deg(p, k, W[0], W[1])
#     true_A = (W != 0).astype(int)
#     scm = sempler.LGANM(W, means, variances)
#     true_means = scm.means
#     true_variances = scm.variances

#     # Build intervention targets
#     true_targets = sempler.generators.intervention_targets(
#         p, e - 1, (0, max_int_size))
#     true_targets = [set()] + [set(I) for I in true_targets]
#     true_targets_union = set.union(*true_targets)
#     interventions = []
#     for targets in true_targets:
#         means = np.random.uniform(i_means[0], i_means[1], len(targets))
#         variances = np.random.uniform(i_vars[0], i_vars[1], len(targets))
#         interventions += [dict(zip(targets, zip(means, variances)))]

#     # Sample data
#     data = []
#     for I in interventions:
#         data.append(scm.sample(n, shift_interventions=I))

#     population_covariances = []
#     for I in interventions:
#         population_covariances.append(scm.sample(population=True, shift_interventions=I).covariance)

#     # Set up score instance
#     gnies_score = Score(data, set(), lmbda=0)
#     gnies_score._sample_covariances = np.array(population_covariances)

#     # gnies_score._data = None

#     data_generating_vars = locals()
#     # Debug outputs
#     print(data_generating_vars)

#     # ------------------------------------------------------
#     # Tests

#     def test_invariance_property(self):
#         # Test that, in the large sample limit, the score improves by
#         # adding to the list of intervention targets a variable whose
#         # noise term distribution is not invariant

#         # We test in the sample limit by using the population
#         # covariances, lmbda = 0, and checking that when a variable's
#         # parents are stable, adding the variable leaves the score
#         # equal (up to a small threshold to account for numerical
#         # issues)

#         # ----------------------------------------
#         # Setup
#         G = NUM_GRAPHS
#         alpha = 0.05
#         run_icp, alpha = False, 0.05
#         # Test
#         I = set()
#         self.gnies_score.set_I(I)
#         for i in range(G):
#             # Generate random graph if i != 0
#             if i == 0:
#                 A = self.true_A
#             else:
#                 A = sempler.generators.dag_avg_deg(self.p, self.k)
#             # Compute score
#             score = self.gnies_score.full_score(A)
#             print("\nGraph %d - I = %s - score score = %0.32f" % (i + 1, I, score))
#             B,_,_ = self.gnies_score._mle_full(A)
#             # Check invariance for all variables
#             for j in range(self.p):
#                 # Compute score resulting from adding j to the list of targets
#                 self.gnies_score.set_I(I | {j})
#                 new_score = self.gnies_score.full_score(A)
#                 delta_score = new_score - score
#                 # Check if parents are a stable set
#                 pa = utils.pa(j, A)
#                 stable = utils.is_stable_set(pa, j, self.true_targets_union, self.W)
#                 stable_str = colored(stable, 'green' if stable else 'red')
#                 # Optionally, run icp to check if set is accepted
#                 if run_icp:
#                     accepted_sets = icp.icp(self.data, j, alpha=alpha, selection=[pa]).accepted
#                     accepted = pa in accepted_sets
#                     accepted_str = colored(accepted, 'green' if accepted else 'red')
#                 else:
#                     accepted_str = 'NA'
#                 # Debug output
#                 print("  %d (stable: %s accepted: %s) ~ %s \t delta score = %0.4f" %
#                       (j, stable_str, accepted_str, pa, delta_score))
#                 # Test
#                 passes = np.isclose(new_score, score, atol=1e-4) if stable else new_score > score
#                 # If test fails, store all information for analysis
#                 store_invariance_result(self.data_generating_vars, locals()) if not passes else None
#                 # Repetitive, but for informative output when running tests
#                 if stable:
#                     self.assertTrue(np.isclose(new_score, score, atol=1e-4))
#                 else:
#                     self.assertGreater(new_score, score)


# def store_invariance_result(data_generating_variables, local_variables):
#     # Remove 'self' attribute as it cannot be pickled
#     to_store = list(local_variables.keys())
#     to_store.remove('self')
#     # Save data generating variables and local variables at moment of failure
#     state = [data_generating_variables] + [dict((k, local_variables[k]) for k in to_store)]
#     # Pickle
#     filename = '%d_error_report:test_invariance_property.pickle' % time.time()
#     with open(filename, 'wb') as f:
#         pickle.dump(state, f)
#     print("Wrote relevant variables to: %s" % filename)
