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

import gnies.scores.interventional as interventional
from gnies.scores import InterventionalScore
from gnies.scores.gnies_score import GnIESScore as FixedInterventionalScore

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
            full_scores, local_scores = score_graphs(graphs_to_score, targets_to_score, datasets)
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
        computed_full_scores, computed_local_scores, mask = score_graphs(graphs, targets, datasets)
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
        _, current_scores,_ = score_graphs(graphs, [targets[0]], datasets)
        for i in range(1, len(targets)):
            change = (targets[i-1] - targets[i]) | (targets[i] - targets[i-1])
            constant = set(range(p)) - change
            print("Comparing I = %s vs I = %s - change = %s" % (targets[i], targets[i-1], change))
            _,next_scores,_ = score_graphs(graphs, [targets[i]], datasets)
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


def score_graphs(graphs, targets, datasets, debug=False):
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
            score = FixedInterventionalScore(data, I)
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
