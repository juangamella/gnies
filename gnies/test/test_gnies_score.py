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
import time
import os

import gnies.scores.gnies_score as gnies_score
from gnies.scores import FixedInterventionalScore
from gnies.scores.gnies_score import GnIESScore

# ---------------------------------------------------------------------
# Tests for the l0-penalized scores

# Number of random graphs generated for each test that uses random graphs
NUM_GRAPHS = 2


class ImplementationChangeTests(unittest.TestCase):
    """Test whether the actual values computed by the score change"""

    datasets_file = 'datasets.npy'
    graphs_file = 'graphs.npy'
    targets_file = 'targets.npy'
    full_scores_file = 'full_scores.npy'
    local_scores_file = 'local_scores.npy'

    def setUp(self):

        # Check if files have already been generated
        files_exist = os.path.exists(self.datasets_file)
        files_exist = files_exist and os.path.exists(self.graphs_file)
        files_exist = files_exist and os.path.exists(self.targets_file)
        files_exist = files_exist and os.path.exists(self.full_scores_file)
        files_exist = files_exist and os.path.exists(self.local_scores_file)

        if files_exist:
            print("Datasets/graphs/scores were already generated")
        if not files_exist:
            print("Datasets/graphs/scores files did not exist; generating")
            datasets = []
            graphs_to_score = []
            targets_to_score = []
            n_datasets = 1
            n_graphs = 10

            # Generate datasets
            n = 1000
            p = 10
            k = 2.3
            envs = 4
            for i in range(n_datasets):
                W = sempler.generators.dag_avg_deg(p, k, 0.5, 1, random_state=i)
                scm = sempler.LGANM(W, (0, 1), (1, 2), random_state=i)
                all_targets = sempler.generators.intervention_targets(p, envs - 1, (1, 3), random_state=i)
                data = [scm.sample(n)]
                for targets in all_targets:
                    interventions = dict(
                        (t, (np.random.uniform(1, 2), np.random.uniform(1, 2))) for t in targets)
                    sample = scm.sample(n, noise_interventions=interventions)
                    data.append(sample)
                data = np.array(data)
                # Save dataset and add true graph and targets
                datasets.append(data)
                A = (W != 0).astype(int)
                graphs_to_score.append(A)
                union = set.union(*[set(t) for t in all_targets])
                targets_to_score.append(union)

            # Generate graphs
            graphs_to_score += [sempler.generators.dag_avg_deg(p, k, 1, 1, random_state=j) for j in range(n_graphs)]

            # Store datasets
            datasets = np.array(datasets)
            with open(self.datasets_file, 'wb') as f:
                np.save(f, datasets)
                print('Saved datasets to "%s"' % self.datasets_file)
            # Store graphs
            graphs_to_score = np.array(graphs_to_score)
            with open(self.graphs_file, 'wb') as f:
                np.save(f, graphs_to_score)
                print('Saved graphs to "%s"' % self.graphs_file)
            # Store targets
            targets_to_score = np.array(targets_to_score)
            with open(self.targets_file, 'wb') as f:
                np.save(f, targets_to_score)
                print('Saved targets to "%s"' % self.targets_file)

            # Compute and save scores
            full_scores, local_scores, _ = score_graphs(FixedInterventionalScore, graphs_to_score, targets_to_score, datasets)
            with open(self.full_scores_file, 'wb') as f:
                np.save(f, full_scores)
                print('Saved full_scores to "%s"' % self.full_scores_file)
            with open(self.local_scores_file, 'wb') as f:
                np.save(f, local_scores)
                print('Saved local_scores to "%s"' % self.local_scores_file)

    def test_scores(self):
        debug = True
        # Load files
        graphs = np.load(self.graphs_file)
        targets = np.load(self.targets_file, allow_pickle=True)
        datasets = np.load(self.datasets_file)
        computed_full_scores, computed_local_scores, mask = score_graphs(GnIESScore, graphs, targets, datasets)
        full_scores = np.load(self.full_scores_file)
        local_scores = np.load(self.local_scores_file)
        diff_full = (computed_full_scores - full_scores)
        print("Diff. in full scores - min :", diff_full.min(), "max :", diff_full.max())
        # print(diff_full)
        diff_local = (computed_local_scores - local_scores)
        print("Diff. in local scores - min :", diff_local.min(), "max :", diff_local.max())
        # print(diff_local)
        # print(diff_local[mask])
        thresh = 1e-11
        #self.assertLess(abs(diff_full).max(), thresh)
        self.assertLess(abs(diff_local).max(), thresh)
        # Test that sum of local scores is full score
        diffs = abs(computed_full_scores - computed_local_scores.sum(axis=3))
        self.assertLess(diffs.max(), 1e-10)

    def test_I_change(self):
        """Check that the local scores remain the same when variables are not added to list of intervention targets"""
        graphs = np.load(self.graphs_file)
        p = graphs.shape[1]
        datasets = np.load(self.datasets_file)
        targets = [set(),
                   {1},
                   {1,2},
                   {1,2,3,4},
                   {1,2,3,4,5},
                   {3,4,5},
                   {4,5},
                   {4}]
        _, current_scores,_ = score_graphs(GnIESScore, graphs, [targets[0]], datasets)
        for i in range(1, len(targets)):
            change = (targets[i-1] - targets[i]) | (targets[i] - targets[i-1])
            constant = set(range(p)) - change
            print("Comparing I = %s vs I = %s - change = %s" % (targets[i], targets[i-1], change))
            _,next_scores,_ = score_graphs(GnIESScore, graphs, [targets[i]], datasets)
            print("Variables which changed")
            for j in change:
                diff = abs(next_scores[:,:,:,j] - current_scores[:,:,:,j])
                print("  %d - max diff." % j, diff.max())
                self.assertTrue((next_scores[:,:,:,j] != current_scores[:,:,:,j]).all())
            print("Variables which didn't change")
            for j in constant:
                diff = abs(next_scores[:,:,:,j] - current_scores[:,:,:,j])
                print("  %d - max diff." % j, diff.max())
                self.assertTrue((next_scores[:,:,:,j] == current_scores[:,:,:,j]).all())
            current_scores = next_scores


def score_graphs(score_class, graphs, targets, datasets, debug=False):
    # Set up score arrays
    p = graphs.shape[1]
    full_scores = np.zeros((len(graphs), len(targets), len(datasets)), dtype=float)
    local_scores = np.zeros((len(graphs), len(targets), len(datasets), p), dtype=float)
    mask = np.zeros((len(graphs), len(targets), len(datasets), p), dtype=bool)
    # Compute scores
    start = time.time()
    for k, data in enumerate(datasets):
        for j, I in enumerate(targets):
            print("Computing scores for dataset %d and targets %s" % (k + 1, targets)) if debug else None
            score = score_class(data, I)
            for i, A in enumerate(graphs):
                print("  Graph", i+1) if debug else None
                full_score = score.full_score(A)
                full_scores[i, j, k] = full_score
                print("   full score :", full_score) if debug else None
                print("   local scores :") if debug else None
                sum_of_locals = 0
                for h in range(p):
                    pa = utils.pa(h, A)
                    local_score = score.local_score(h, pa)
                    sum_of_locals += local_score
                    local_scores[i, j, k, h] = local_score
                    mask[i, j, k, h] = (h not in I) or (len(pa) == 0)
                    print("    %d : " % h, local_score) if debug else None
                print("full :", full_score, "sum :", sum_of_locals) if debug else None
                print() if debug else None
    print("Scored graphs in %0.2f seconds" % (time.time() - start))
    return full_scores, local_scores, mask


class ScoreTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    rng = np.random.default_rng(42)
    true_B = true_A * rng.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4), random_state=42)
    p = len(true_A)
    n = 100000
    true_targets = [set(), {0}, {1}, {2}, {3}, {4}]
    interventions = [{0: (1.1, 1.0)},
                     {1: (2.2, 1.1)},
                     {2: (3.3, 1.2)},
                     {3: (4.4, 1.3)},
                     {4: (5.5, 1.4)}]
    obs_data = scm.sample(n=n, random_state=42)
    int_data = [obs_data]
    n_obs = [n] * (len(interventions) + 1)
    e = len(interventions) + 1
    # Sample interventional distributions and construct true interventional
    # variances for later reference in tests
    interventional_variances = np.tile(scm.variances, (len(interventions) + 1, 1))
    interventional_means = np.tile(scm.means, (len(interventions) + 1, 1))
    for i, intervention in enumerate(interventions):
        int_data.append(scm.sample(n=n, shift_interventions=intervention, random_state=42))
        for (target, params) in intervention.items():
            interventional_variances[i + 1, target] += params[1]
            interventional_means[i + 1, target] += params[0]
    # Set up score instantces
    I = np.eye(scm.p)
    print("Sample size =", n)
    print("True noise means:\n", interventional_means)
    print("True noise variances:\n", interventional_variances)
    print("True connectivity:\n", scm.W)

    # ------------------------------------------------------
    # White-box tests:
    #   testing the inner workings of the gnies.scores.interventional module, e.g. the
    #   intermediate functions used to compute the likelihoods

    def test_ddof(self):
        A = np.zeros((self.p, self.p))
        # If no interventions are done, there should be p variances,
        # no matter the number of environments
        self.assertEqual(self.p, gnies_score.ddof_full(A, set(), 1))
        self.assertEqual(self.p, gnies_score.ddof_full(A, set(), 2))
        self.assertEqual(self.p, gnies_score.ddof_full(A, set(), 3))
        # If intercept is used
        self.assertEqual(self.p * 2, gnies_score.ddof_full(A, set(), 3, False))
        # A few more
        e = 3
        I = {1,2,4}
        # 1 for x=0, 3 for x=1, 3 for x=2, 1 for x=3, 3 for x=4
        # i.e a total of 1 + 3 + 3 + 1 + 3 = 11
        self.assertEqual(11, gnies_score.ddof_full(A, I, e))
        self.assertEqual(22, gnies_score.ddof_full(A, I, e, False))

    def test_ddof_decomposability(self):
        G = NUM_GRAPHS
        p = 10
        e = 5
        rng = np.random.default_rng(42)
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1, random_state=i)
            size = rng.integers(0,p)
            I = set(np.random.choice(range(p), size))
            # Without intercept
            full_ddof = gnies_score.ddof_full(A, I, e)
            acc = 0
            for j in range(p):
                acc += gnies_score.ddof_local(j, utils.pa(j, A), I, e)
            self.assertEqual(full_ddof, acc)
            # With intercept
            full_ddof = gnies_score.ddof_full(A, I, e, False)
            acc = 0
            for j in range(p):
                acc += gnies_score.ddof_local(j, utils.pa(j, A), I, e, False)
            self.assertEqual(full_ddof, acc)

    def test_score_decomposability_coarse(self):
        # The score should respect decomposability (coarse grained)
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        K = 5
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # Generate random intervention targets
        random_interventions = sempler.generators.intervention_targets(self.p,
                                                                       K,
                                                                       (0, self.p), random_state=42)
        interventions = [set.union(*self.true_targets)] + [set(I) for I in random_interventions]
        # ----------------------------------------
        # Test that score is the same when computed locally or for
        # the whole graph
        for centered in [True, False]:
            for I in interventions:
                score = GnIESScore(self.int_data, I, centered=centered)
                for A in graphs:
                    print("\n\nInterventions set (coarse): ", I)
                    # Full score
                    full_score = score.full_score(A)
                    # Sum of local scores
                    acc = 0
                    for j in range(self.p):
                        pa = np.where(A[:, j] != 0)[0]
                        local_score = score.local_score(j, pa)
                        print("  ", j, pa, local_score)
                        acc += local_score
                    print("Full vs. acc:", full_score, acc)
                    self.assertTrue(np.isclose(full_score, acc))

    def test_mle_1_centered(self):
        # Check that ML estimates are reasonably close to the true model
        I = set.union(*self.true_targets)
        score = GnIESScore(self.int_data, I, lmbda=None)
        # Full MLE
        B_full, omegas_full, means_full = score._mle_full(self.true_A)
        self.assertIsNone(means_full)
        print("FULL")
        print(B_full)
        print(omegas_full)
        # Local MLE
        B_local, omegas_local = np.zeros_like(B_full), np.zeros_like(omegas_full)
        for (j, pa) in self.factorization:
            B_local[:, j], omegas_local[:, j], means = score._mle_local(j, pa)
            self.assertIsNone(means)
        print("LOCAL")
        print(B_local)
        print(omegas_local)
        # Test that the estimates are reasonably close to the true model
        self.assertTrue(np.allclose(self.interventional_variances, omegas_local, atol=1e-1))
        self.assertTrue(np.allclose(self.interventional_variances, omegas_full, atol=1e-1))
        self.assertTrue(np.allclose(self.true_B, B_local, atol=5e-2))
        self.assertTrue(np.allclose(self.true_B, B_full, atol=5e-2))

    def test_mle_1_uncentered(self):
        # Check that ML estimates are reasonably close to the true model
        I = set.union(*self.true_targets)
        score = GnIESScore(self.int_data, I, centered=False, lmbda=None)
        # Full MLE
        B_full, omegas_full, nus_full = score._mle_full(self.true_A)
        print("FULL")
        print(B_full)
        print(nus_full)
        print(omegas_full)
        # Local MLE
        B_local = np.zeros_like(B_full)
        omegas_local = np.zeros_like(omegas_full)
        nus_local = np.zeros((self.e, self.p))
        for (j, pa) in self.factorization:
            B_local[:, j], omegas_local[:, j], nus_local[:, j] = score._mle_local(j, pa)
        print("LOCAL")
        print(B_local)
        print(nus_local)
        print(omegas_local)
        print(self.interventional_means)
        # Test that the estimates are reasonably close to the true model
        self.assertTrue(np.allclose(self.interventional_means, nus_full, atol=1e-1))
        self.assertTrue(np.allclose(self.interventional_means, nus_local, atol=1e-1))
        self.assertTrue(np.allclose(self.interventional_variances, omegas_full, atol=1e-1))
        self.assertTrue(np.allclose(self.interventional_variances, omegas_local, atol=1e-1))
        self.assertTrue(np.allclose(self.true_B, B_full, atol=5e-2))
        self.assertTrue(np.allclose(self.true_B, B_local, atol=5e-2))

    def test_mle_2_centered(self):
        # Check that MLE works properly and respects restriction on
        # noise term variances imposed by I

        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        K = 5
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # Generate random intervention targets
        interventions = [self.true_targets]
        for _ in range(K):
            random_interventions = sempler.generators.intervention_targets(self.p,
                                                                           self.e,
                                                                           (0, self.p), random_state=42)
            interventions.append(set.union(*[set(I) for I in random_interventions]))

        # ----------------------------------------
        # Test behaviour of MLE
        start = time.time()
        for I in interventions:
            score = GnIESScore(self.int_data, I, lmbda=None)
            for A in graphs:
                # print("\n\nInterventions set: ", I)
                # Full MLE
                B_full, omegas_full, means = score._mle_full(A)
                self.assertIsNone(means)
                # print("FULL")
                # print(B_full)
                # print(omegas_full)
                # Local MLE
                B_local, omegas_local = np.zeros_like(B_full), np.zeros_like(omegas_full)
                for j in range(self.p):
                    pa = np.where(A[:, j] != 0)[0]
                    B_local[:, j], omegas_local[:, j], means = score._mle_local(j, pa)
                    self.assertIsNone(means)
                # print("LOCAL")
                # print(B_local)
                # print(omegas_local)
                # Test that result is the same when parameters are
                # estimated locally or using the full graph
                self.assertTrue((B_full == B_local).all())
                self.assertTrue((omegas_full == omegas_local).all())
                # Test that the constrainst imposed by the intervention
                # targets hold
                for j in range(self.p):
                    if j not in I:
                        self.assertEqual(1, len(np.unique(omegas_local[:,j])))
                        self.assertEqual(1, len(np.unique(omegas_full[:,j])))
                    else:
                        self.assertEqual(self.e, len(np.unique(omegas_local[:,j])))
                        self.assertEqual(self.e, len(np.unique(omegas_full[:,j])))
                    # Check DDOF
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(omegas_local[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(omegas_full[:, j])))
        print("Tested MLE behaviour (centered) for %d cases (%d graphs x %d intervention sets) (%0.2f s)" %
              ((G + 1) * (K + 1), G + 1, K + 1, time.time() - start))

    def test_mle_2_uncentered(self):
        # Check that MLE works properly and respects restriction on
        # noise term variances imposed by I

        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        K = 5
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # Generate random intervention targets
        interventions = [self.true_targets]
        for _ in range(K):
            random_interventions = sempler.generators.intervention_targets(self.p,
                                                                           self.e,
                                                                           (0, self.p), random_state=42)
            interventions.append(set.union(*[set(I) for I in random_interventions]))

        # ----------------------------------------
        # Test behaviour of MLE
        start = time.time()
        for I in interventions:
            score = GnIESScore(self.int_data, I, centered=False, lmbda=None)
            for A in graphs:
                # print("\n\nInterventions set: ", I)
                # Full MLE
                B_full, omegas_full, means_full = score._mle_full(A)
                # print("FULL")
                # print(B_full)
                # print(omegas_full)
                # Local MLE
                B_local, omegas_local, means_local = np.zeros_like(B_full), np.zeros_like(omegas_full), np.zeros_like(means_full)
                for j in range(self.p):
                    pa = np.where(A[:, j] != 0)[0]
                    B_local[:, j], omegas_local[:, j], means_local[:,j] = score._mle_local(j, pa)
                # print("LOCAL")
                # print(B_local)
                # print(omegas_local)
                # Test that result is the same when parameters are
                # estimated locally or using the full graph
                self.assertTrue((B_full == B_local).all())
                self.assertTrue((omegas_full == omegas_local).all())
                self.assertTrue((means_full == means_local).all())
                # Test that the constrainst imposed by the intervention
                # targets hold
                for j in range(self.p):
                    if j not in I:
                        self.assertEqual(1, len(np.unique(omegas_local[:,j])))
                        self.assertEqual(1, len(np.unique(omegas_full[:,j])))
                        self.assertEqual(1, len(np.unique(means_local[:,j])))
                        self.assertEqual(1, len(np.unique(means_full[:,j])))
                    else:
                        self.assertEqual(self.e, len(np.unique(omegas_local[:,j])))
                        self.assertEqual(self.e, len(np.unique(omegas_full[:,j])))
                        self.assertEqual(self.e, len(np.unique(means_local[:,j])))
                        self.assertEqual(self.e, len(np.unique(means_full[:,j])))
                    # Check DDOF
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(omegas_local[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(omegas_full[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(means_local[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e),
                                     len(np.unique(means_full[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e, centered=False),
                                     len(np.unique(means_local[:, j])) + len(np.unique(omegas_local[:, j])))
                    self.assertEqual(gnies_score.ddof_local(j, set(), I, self.e, centered=False),
                                     len(np.unique(means_full[:, j])) + len(np.unique(means_full[:, j])))
        print("Tested MLE behaviour (centered) for %d cases (%d graphs x %d intervention sets) (%0.2f s)" %
              ((G + 1) * (K + 1), G + 1, K + 1, time.time() - start))

    def test_mle_means_1(self):
        # If all variables are intervened in all environments,
        # (I-B)^-1 @ nus should equal the sample means
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # ----------------------------------------
        # Test
        true_sample_means = np.array([np.mean(X, axis=0) for X in self.int_data])
        score = GnIESScore(self.int_data, set(range(self.p)), centered=False)
        for A in graphs:
            B, _, noise_term_means = score._mle_full(A)
            sample_means = noise_term_means @ np.linalg.inv(np.eye(self.p) - B)
            # print(true_sample_means)
            # print(sample_means)
            self.assertTrue(np.allclose(true_sample_means, sample_means))

    def test_mle_means_2(self):
        # If there are no interventions (I-B)^-1 @ nus should equal
        # the pooled means
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        # Compute the pooled means
        assert len(np.unique(self.n_obs) == 1)
        sample_means = np.array([np.mean(X, axis=0) for X in self.int_data])
        pooled_means = np.tile(sample_means.mean(axis=0), (self.e, 1))
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # ----------------------------------------
        # Test
        score = GnIESScore(self.int_data, set(), centered=False)
        for A in graphs:
            B, _, noise_term_means = score._mle_full(A)
            sample_means = noise_term_means @ np.linalg.inv(np.eye(self.p) - B)
            # print(pooled_means)
            # print(sample_means)
            self.assertTrue(np.allclose(pooled_means, sample_means))

    def test_mle_means_3(self):
        # If a variable has no parents and is intervened, (I-B)^-1 @
        # nus should equal the sample mean for that variable in that
        # environments
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # ----------------------------------------
        # Test
        true_sample_means = np.array([np.mean(X, axis=0) for X in self.int_data])
        for A in graphs:
            for j in range(self.p):
                pa = list(np.where(A[:, j] != 0)[0])
                if len(pa) > 0:
                    continue
                score = GnIESScore(self.int_data, {j}, centered=False)
                B, _, noise_term_means = score._mle_full(A)
                sample_means = noise_term_means @ np.linalg.inv(np.eye(self.p) - B)
                self.assertTrue(np.allclose(true_sample_means[:, j], sample_means[:, j]))

    def test_centered_vs_uncentered(self):
        # The centered score always uses the MLE of the means
        # (implicitly by using the sample covariance for the
        # likelihood computation). Check that when all variables
        # receive interventions and lmbda = 0, both scores match.
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS
        k = 2.1
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        I = set(range(self.p))
        for lmbda in [0, None]:
            centered = GnIESScore(self.int_data, I, lmbda=lmbda, centered=True)
            uncentered = GnIESScore(self.int_data, I, lmbda=lmbda, centered=False)
            self.assertTrue(centered.centered)
            self.assertFalse(uncentered.centered)
            for A in graphs:
                # Check full score
                full_centered = centered.full_score(A)
                full_uncentered = uncentered.full_score(A)
                if lmbda == 0:
                    self.assertTrue(np.isclose(full_centered, full_uncentered))
                else:
                    self.assertFalse(np.isclose(full_centered, full_uncentered))
                # Check local scores
                for j in range(self.p):
                    pa = np.where(A[:, j] != 0)[0]
                    local_centered = centered.local_score(j, pa)
                    local_uncentered = uncentered.local_score(j, pa)
                    if lmbda == 0:
                        self.assertTrue(np.isclose(local_centered, local_uncentered))
                    else:
                        self.assertFalse(np.isclose(local_centered, local_uncentered))

    def test_model_complexity_1(self):
        # Test that adding intervention targets always increases the
        # score when lambda=0
        # ----------------------------------------
        # Setup
        G = NUM_GRAPHS * 10
        k = 2.1
        runs = 10
        # Generate random graphs
        graphs = [self.true_A]
        graphs += [sempler.generators.dag_avg_deg(self.p, k, random_state=i) for i in range(G)]
        # ----------------------------------------
        # Test
        for centered in [True, False]:
            I = set()
            score = GnIESScore(self.int_data, I, lmbda=0, centered=centered)
            for k, A in enumerate(graphs):
                print("Graph", k)
                current_score = score.full_score(A)
                for r in range(self.p):
                    print("  ", I)
                    # If the intervention targets are already all
                    # targets, stop
                    if len(I) == self.p:
                        break
                    # Build new intervention list
                    new_I = I | {r}
                    score.set_I(new_I)
                    # Compute new score
                    new_score = score.full_score(A)
                    if new_score < current_score:
                        print("-------------------")
                        print("centered =", centered, I, new_I)
                        self.assertGreaterEqual(new_score, current_score)
                    current_score, I = new_score, new_I

    # def test_model_complexity_2(self):
    #     # Test that adding edges to a graph always increases the score
    #     # when lambda=0
    #     # ----------------------------------------
    #     # Setup
    #     G = NUM_GRAPHS * 10
    #     k = 2.1
    #     runs = 10
    #     # Generate random graphs
    #     graphs = [self.true_A]
    #     graphs += [sempler.generators.dag_avg_deg(self.p, k) for _ in range(G)]
    #     # ----------------------------------------
    #     # Test
    #     for centered in [True, False]:
    #         score = InterventionalScore(self.int_data, lmbda=0, debug=0,
    #                                   centered=centered, max_iter=500)
    #         for k, A in enumerate(graphs):
    #             print("Graph", k)
    #             random_targets = sempler.generators.intervention_targets(self.p, self.e, (0, self.p))
    #             I = [set(i) for i in random_targets]
    #             I = [set()] * self.e
    #             current_score = score.full_score(A, I)
    #             to,fro = np.where(A == 0)
    #             candidate_edges = list(zip(to,fro))
    #             for _ in range(runs):
    #                 print("  %d edges" % A.sum())
    #                 if len(candidate_edges) == 0:
    #                     print("  Ran out of possible edges.")
    #                     break
    #                 to,fro = candidate_edges[0]
    #                 candidate_edges = candidate_edges[1:]
    #                 # Build new graph
    #                 new_A = A.copy()
    #                 new_A[to,fro] = 1
    #                 if not utils.is_dag(new_A):
    #                     continue
    #                 # Compute new score
    #                 new_score = score.full_score(new_A, I)
    #                 if new_score < current_score:
    #                     print("-------------------")
    #                     print("centered =", centered, A, new_A)
    #                     self.assertGreaterEqual(new_score, current_score)
    #                 current_score, A = new_score, new_A
