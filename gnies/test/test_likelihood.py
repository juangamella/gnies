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
import time
import gnies.scores.log_likelihood_means as log_likelihood
from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen

# ---------------------------------------------------------------------
# Tests for the l0-penalized scores

NUM_GRAPHS = 100
COMPARE_RAW = False


def full_score(B, means, variances, XX):
    score = log_likelihood.full(B, means, variances, XX)
    if COMPARE_RAW:
        score_raw = log_likelihood.full_raw(B, means, variances, XX)
        assert np.isclose(score, score_raw)
    return score


def mle_noise_distribution(B, XX):
    """Given data and a connectivity, return the MLE estimates for the
    noise term means and variances"""
    sample_covariances = [np.cov(X, rowvar=False, ddof=0) for X in XX]
    sample_means = np.mean(XX, axis=1)
    I_B = (np.eye(len(B)) - B.T)
    means_mle = np.array([I_B @ sample_mean for sample_mean in sample_means])
    variances_mle = np.array([np.diag(I_B @ sigma @ I_B.T) for sigma in sample_covariances])
    return means_mle, variances_mle


class ScoreTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (-1, 1), (0.3, 0.4))
    p = len(true_A)
    n = 10000
    interventions = [{0: (1.1, 1.0)},
                     {1: (2.1, 1.1)},
                     {2: (-3.1, 1.2)},
                     {3: (-1.7, 1.3)},
                     {4: (-1.5, 1.4)}]
    obs_data = scm.sample(n=n)
    int_data = [obs_data]
    n_obs = [n] * (len(interventions) + 1)
    e = len(interventions) + 1
    # Sample interventional distributions and construct true interventional
    # variances for later reference in tests
    interventional_variances = np.tile(scm.variances, (len(interventions) + 1, 1))
    interventional_means = np.tile(scm.means, (len(interventions) + 1, 1))
    for i, intervention in enumerate(interventions):
        int_data.append(scm.sample(n=n, shift_interventions=intervention))
        for (target, params) in intervention.items():
            interventional_variances[i + 1, target] += params[1]
            interventional_means[i + 1, target] += params[0]
    # Set up score instantces
    obs_score = GaussObsL0Pen(obs_data)
    I = np.eye(scm.p)
    print("True noise means:\n", interventional_means)
    print("True noise variances:\n", interventional_variances)

    # ------------------------------------------------------
    # White-box tests:
    #   testing the inner workings of the ges.scores module, e.g. the
    #   intermediate functions used to compute the likelihoods

    def test_likelihood_obs(self):
        # As a white-box test, make sure the true model has higher
        # likelihood than the empty graph
        # Compute likelihood of the true model
        score_true_model = full_score(self.true_B,
                                      np.atleast_2d(self.scm.means),
                                      np.atleast_2d(self.scm.variances),
                                      [self.obs_data])
        # Compute likelihood of the empty model
        marginal_variances = np.diag(self.scm.sample(population=True).covariance)
        marginal_means = np.linalg.inv(self.I - self.scm.W) @ self.scm.means
        score_empty_model = full_score(np.zeros_like(self.true_B),
                                       np.atleast_2d(marginal_means),
                                       np.atleast_2d(marginal_variances),
                                       [self.obs_data])
        self.assertGreater(score_true_model, score_empty_model)

    def test_likelihood_int(self):
        # As a white-box test, make sure the true model has higher
        # likelihood than the empty graph
        # Compute likelihood of the true model
        score_true_model = full_score(self.true_B,
                                      self.interventional_means,
                                      self.interventional_variances,
                                      self.int_data)
        # Compute likelihood of the empty model
        marginal_variances = [np.diag(self.scm.sample(population=True).covariance)]
        marginal_means = [self.scm.sample(population=True).mean]
        for i in self.interventions:
            marginal_variances.append(np.diag(self.scm.sample(
                population=True, shift_interventions=i).covariance))
            marginal_means.append(self.scm.sample(population=True, shift_interventions=i).mean)
        marginal_variances = np.array(marginal_variances)
        marginal_means = np.array(marginal_means)
        assert marginal_variances.shape == (len(self.int_data), self.p)
        score_empty_model = full_score(np.zeros_like(self.true_B),
                                       marginal_means,
                                       marginal_variances,
                                       self.int_data)
        self.assertGreater(score_true_model, score_empty_model)

    def test_likelihood_decomposability_obs(self):
        # As a white-box test, make sure the likelihood functions
        # preserve decomposability
        print("Decomposability of observational likelihood")
        score_full = full_score(
            self.true_B,
            np.atleast_2d(self.scm.means),
            np.atleast_2d(self.scm.variances),
            [self.obs_data])
        acc = 0
        for (j, pa) in self.factorization:
            score_local = log_likelihood.local_raw(j,
                                                   self.true_B[:, j],
                                                   np.atleast_2d(self.scm.means[j]),
                                                   np.atleast_1d(self.scm.variances[j]),
                                                   [self.obs_data])
            print("  ", j, pa, score_local)
            acc += score_local
        print("Full vs. acc:", score_full, acc)
        self.assertAlmostEqual(score_full, acc, places=2)

    def test_likelihood_decomposability_int(self):
        # As a white-box test, make sure the likelihood functions
        # preserve decomposability
        print("Decomposability of interventional likelihood")
        # Compute score with the full model
        score_full = full_score(
            self.true_B,
            self.interventional_means,
            self.interventional_variances,
            self.int_data)
        acc = 0
        for (j, pa) in self.factorization:
            score_local = log_likelihood.local_raw(
                j, self.true_B[:, j],
                self.interventional_means[:, j],
                self.interventional_variances[:, j],
                self.int_data)
            print("  ", j, pa, score_local)
            acc += score_local
        print("Full vs. acc:", score_full, acc)
        self.assertAlmostEqual(score_full, acc, places=2)

    # ------------------------------------------------------
    # Black-box tests:
    #   Testing the behaviour of the "API" functions, i.e. the
    #   functions to compute the full/local
    #   observational/interventional BIC scores from a given DAG
    #   structure and the data

    def test_vs_obs_score_true_graph(self):
        # Test that the likelihood on a single sample is
        # very close to the observational score
        obs_score = GaussObsL0Pen(self.obs_data, lmbda=0).full_score(self.true_A)
        B, omegas = GaussObsL0Pen(self.obs_data)._mle_full(self.true_A)
        # Compute the MLE of the means
        means = (self.I - B.T) @ np.mean(self.obs_data, axis=0)
        likelihood = full_score(B, [means], [omegas], [self.obs_data])
        print("obs/int", obs_score, likelihood)
        self.assertAlmostEqual(obs_score, likelihood)

    def test_mle_is_max_obs(self):
        # Check that for any graph, the MLE estimates of the noise
        # term means and variances are in fact maximal
        start = time.time()
        G = NUM_GRAPHS
        cases = 5
        k = 2.1
        for i in range(G):
            W = sempler.generators.dag_avg_deg(self.p, k, w_min=0.1, w_max=3)
            # Compute MLEs and their score
            means_mle, variances_mle = mle_noise_distribution(W, [self.obs_data])
            # print("MLEs", means_mle, variances_mle)
            score_mle = full_score(W, means_mle, variances_mle, [self.obs_data])
            # Build perturbed estimates
            perturbed_variances = variances_mle + \
                np.random.uniform(0, 0.01, size=(cases, 1, self.p))
            perturbed_means = means_mle + \
                np.random.uniform(0, 0.01, size=(cases, 1, self.p))
            for (means, variances) in zip(perturbed_means, perturbed_variances):
                # print("Perturbed estimate:", means, variances)
                score = full_score(W, means, variances, [self.obs_data])
                self.assertGreater(score_mle, score)
        end = time.time()
        print("Checked MLE maximality for %d graphs and %d perturbations (envs:1, elapsed:%0.2f s)" % (
            i + 1, cases, end - start))

    def test_mle_is_max_int(self):
        # Check that for any graph, the MLE estimates of the noise
        # term means and variances are in fact maximal
        G = NUM_GRAPHS
        cases = 5
        k = 2.1
        start = time.time()
        for i in range(G):
            W = sempler.generators.dag_avg_deg(self.p, k, w_min=0.1, w_max=3)
            # Compute MLEs and their score
            means_mle, variances_mle = mle_noise_distribution(W, self.int_data)
            # print("MLEs", means_mle, variances_mle)
            score_mle = full_score(W, means_mle, variances_mle, self.int_data)
            # Build perturbed estimates
            perturbed_variances = variances_mle + \
                np.random.uniform(0, 0.01, size=(cases, self.e, self.p))
            perturbed_means = means_mle + \
                np.random.uniform(0, 0.01, size=(cases, self.e, self.p))
            for (means, variances) in zip(perturbed_means, perturbed_variances):
                # print("Perturbed estimate:", means, variances)
                score = full_score(W, means, variances, self.int_data)
                self.assertGreater(score_mle, score)
        end = time.time()
        print("Checked MLE maximality for %d graphs and %d perturbations (envs:%d, elapsed:%0.2f s)" % (
            i + 1, cases, self.e, end - start))

    def test_mle_vs_true_vs_empty(self):
        # True connectivity + MLE estimates
        means_mle, variances_mle = mle_noise_distribution(self.scm.W, self.int_data)
        score_true_mle = full_score(
            self.scm.W, means_mle, variances_mle, self.int_data)
        # True connectivity + true parameters
        score_true = full_score(
            self.scm.W, self.interventional_means, self.interventional_variances, self.int_data)
        # Empty graph
        means_empty, variances_empty = mle_noise_distribution(
            np.zeros_like(self.scm.W), self.int_data)
        score_empty = full_score(
            self.scm.W, means_empty, variances_empty, self.int_data)
        # Test
        self.assertGreater(score_true_mle, score_true)
        self.assertGreater(score_true, score_empty)

    def test_decomposability_int_1(self):
        # Check that for any graph, the sum of the local scores
        # (according to the graph's factorization) are equal to the
        # global score. (With MLE estimates)
        start = time.time()
        G = NUM_GRAPHS
        k = 2.1
        for i in range(G):
            W = sempler.generators.dag_avg_deg(self.p, k, w_min=0.1, w_max=3)
            # Compute MLEs
            means_mle, variances_mle = mle_noise_distribution(W, self.int_data)
            # print("Parameters (MLEs)", means_mle, variances_mle)
            score_full = full_score(W, means_mle, variances_mle, self.int_data)
            # Compute sum of local scores
            score_local = 0
            for j in range(self.p):
                score_local += log_likelihood.local_raw(j,
                                                        W[:, j],
                                                        means_mle[:, j],
                                                        variances_mle[:, j],
                                                        self.int_data)
            # Test
            self.assertTrue(np.isclose(score_local, score_full))
            end = time.time()
        print("Checked decomposability for %d graphs (e: %d, p: %d, elapsed:%0.2f s)" % (
            i + 1, self.e, self.p, end - start))

    def test_decomposability_int_2(self):
        # Check that for any graph, the sum of the local scores
        # (according to the graph's factorization) are equal to the
        # global score. (With MLE estimates)
        start = time.time()
        G = NUM_GRAPHS
        k = 2.1
        for i in range(G):
            W = sempler.generators.dag_avg_deg(self.p, k, w_min=0.1, w_max=3)
            # Set random parameters
            means = np.random.uniform(-1, 1, size=(self.e, self.p))
            variances = np.random.uniform(1, 2, size=(self.e, self.p))
            # print("Parameters", means, variances)
            score_full = full_score(W, means, variances, self.int_data)
            # Compute sum of local scores
            score_local = 0
            for j in range(self.p):
                score_local += log_likelihood.local_raw(j,
                                                        W[:, j],
                                                        means[:, j],
                                                        variances[:, j],
                                                        self.int_data)
            # Test
            self.assertTrue(np.isclose(score_local, score_full))
            end = time.time()
        print("Checked decomposability for %d graphs (e: %d, p: %d, elapsed:%0.2f s)" % (
            i + 1, self.e, self.p, end - start))

    def test_vs_obs_score_random_graphs(self):
        # Check that for any graph, the score with the MLE estimate
        # matches that of GaussObsL0Pen.
        start = time.time()
        G = NUM_GRAPHS * 5
        k = 2.1
        for i in range(G):
            A = sempler.generators.dag_avg_deg(self.p, k, w_min=1, w_max=1)
            # Estimate B
            obs_score = GaussObsL0Pen(self.obs_data, lmbda=0)
            B, omegas = obs_score._mle_full(A)
            score_obs = obs_score.full_score(A)
            # Compute MLEs & score
            means_mle, variances_mle = mle_noise_distribution(B, [self.obs_data])
            # print("Parameters (MLEs)", means_mle, variances_mle)
            score_lik = full_score(B, means_mle, variances_mle, [self.obs_data])
            # Test
            print("obs/lik", score_obs, score_lik)
            self.assertTrue(np.isclose(score_obs, score_lik))
            end = time.time()
        print("Checked vs. obs. score for %d graphs (elapsed:%0.2f s)" % (i + 1, end - start))
