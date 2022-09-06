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

"""This module contains the implementation of the GnIES score (see
"Score Function" in section 4 of the GnIES paper). The score is
implemented by the FixedInterventionalScore class.

GnIES paper: <TODO: arxiv link>

"""

import numpy as np
from . import log_likelihood
from . import log_likelihood_means
from .decomposable_score import DecomposableScore
import gnies.utils as utils

# --------------------------------------------------------------------
# Interventional Score Class (see FixedInterventionalScore below for GnIES score)

class GnIESScore(DecomposableScore):
    """TODO.

    The GnIES score is the score with `fine_grained=False` and is
    implemented in the `FixedInterventionalScore` class, which
    inherits from this one.

    Parameters
    ----------
    e : int
        The number of environments.
    p : int
        The number of variables.
    n_obs : list of ints
        The number of observations from each environment.
    N : int
        The total number of observations from all environments.
    lmbda : float
        The penalization parameter used in the computation of the score.
    max_iter : float
        The maximum number of iterations for the alternating
        optimization procedure.
    tol : float
        The threshold of convergence for the alternating optimization
        procedure; when the maximum distance between elements of
        consecutive Bs is below this threshold, stop the alternating
        optimization and return the latest estimate.
    fine_grained : bool
        How finely we specify intervention targets. `True` means we
        define intervention targets per environment; `False` means we
        allow the noise-term distribution of an intervened variable to
        vary across all environments. The latter is used for the
        computation of the GnIES score.
    centered : bool
        Whether the data is centered before computing the score
        (`centered=True`), or the noise-term means are also estimated
        respecting the constraints imposed by `I`.

    """

    def __init__(self, data, I, centered=True, lmbda=None, tol=1e-16, max_iter=10, cache=True):
        """
        Creates a new instance of the score.

        Parameters
        ----------
        data : list of numpy.ndarray
            A list with the samples from the different environments, where
            each sample is an array with columns corresponding to
            variables and rows to observations.
        centered : bool
            Whether the data is centered before computing the score
            (`centered=True`), or the noise-term means are also estimated
            respecting the constraints imposed by `I`.
        fine_grained : bool
            How finely we specify intervention targets. `True` means we
            define intervention targets per environment; `False` means we
            allow the noise-term distribution of an intervened variable to
            vary across all environments. The latter is used for the
            computation of the GnIES score.
        lmbda : float
            The penalization parameter used in the computation of the score.
        tol : float
            The threshold of convergence for the alternating optimization
            procedure; when the maximum distance between elements of
            consecutive Bs is below this threshold, stop the alternating
            optimization and return the latest estimate.
        max_iter : float
            The maximum number of iterations for the alternating
            optimization procedure.
        """
        super().__init__(data, cache=cache)
        self.I = I.copy()
        self.e = len(data)
        self.p = data[0].shape[1]
        self.n_obs = np.array([len(env) for env in data])
        self.N = sum(self.n_obs)
        self.lmbda = 0.5 * np.log(self.N) if lmbda is None else lmbda
        self.max_iter = max_iter
        self.tol = tol
        self.centered = centered

        # Precompute scatter matrices
        self._sample_covariances = np.array([np.cov(env, rowvar=False, ddof=0) for env in data])
        self._pooled_covariance =  np.sum(self._sample_covariances * np.reshape(self.n_obs, (self.e, 1, 1)), axis=0) / self.N

        # Precompute sample means
        if not centered:
            self._sample_means = np.array([np.mean(env, axis=0) for env in self._data])
            self._pooled_means = np.sum(self._sample_means * np.reshape(self.n_obs, (self.e, 1)), axis=0) / self.N

    def set_I(self, new_I):
        change = (self.I - new_I) | (new_I - self.I)
        self.prune_cache(change)
        self.I = new_I.copy()

    def full_score(self, A):
        """
        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a collection of samples from different environments, by finding
        the maximum likelihood estimates of the corresponding connectivity
        matrix (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j

        Returns
        -------
        score : float
            the penalized log-likelihood score

        """
        # Compute MLE and its likelihood
        if self.centered:
            B, omegas = self._mle_full(A)
            likelihood = log_likelihood.full(B, omegas, self._sample_covariances, self.n_obs)
        else:
            B, nus, omegas = self._mle_full(A)
            likelihood = log_likelihood_means.full(B, nus, omegas, self._data)
        #   Note: the number of parameters is the number of edges +
        #   the total number of marginal variances/means, which depends on
        #   the number of interventions.
        l0_term = self.lmbda * ddof_full(A, self.I, self.e, centered=self.centered)
        score = likelihood - l0_term
        return score

    # def local_score(self, x, pa, I):
    #   already defined in parent class DecomposableScore, which calls _compute_local_score

    def _compute_local_score(self, j, pa):
        """
        Given a node and its parents, return the local l0-penalized
        log-likelihood of a collection of samples from different
        environments, by finding the maximum likelihood estimates of the
        weights and noise term variances.

        Parameters
        ----------
        j : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the penalized log-likelihood score

        """
        # Compute MLE, with the local subgraph p -> x for p in pa
        #   Only do set -> list conversion once, as the ordering is not
        #   guaranteed to be consistent.
        ddof = self.e if j in self.I else 1
        if self.centered:
            b, omegas = self._mle_local(j, pa)
            likelihood = -0.5 * ((1 + np.log(omegas)) * self.n_obs).sum()
        else:
            b, nus, omegas = self._mle_local(j, pa)
            likelihood = log_likelihood_means.local(j, b, nus, omegas, self._data)
        #  Note: the number of parameters is the number of parents (one
        #  weight for each) + one marginal variance/mean per environment for x
        l0_term = self.lmbda * ddof_local(j, pa, self.I, self.e, centered=self.centered)
        score = likelihood - l0_term
        return score

    # --------------------------------------------------------------------
    #  Functions for the maximum likelihood estimation of the
    #  weights/variances
    def _mle_full(self, A):
        """Given the DAG adjacency A and observations, compute the maximum
        likelihood estimate of the connectivity weights and noise
        variances, returning them and the resulting log likelihood.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j

        Returns
        -------
        B : np.array
            the MLE estimate of the connectivity matrix (the weights)
        omegas : np.array
            the MLE estimate of the noise variances of each variable

        """
        B = np.zeros((self.p, self.p), dtype=float)
        Omegas = np.zeros((self.e, self.p), dtype=float)
        for j in range(self.p):
            pa = utils.pa(j, A)
            B[:, j], Omegas[:, j] = self._mle_local(j, pa)
        if self.centered:
            return B, Omegas
        else:
            # noise_term_means = _noise_means_from_B(B, I, self._sample_means, self.n_obs)
            # return B, noise_term_means, omegas
            raise NotImplementedError("Not implemented")

    def _mle_local(self, j, pa):
        pa = sorted(pa)
        b, omegas = self._alternating_mle(j, pa, debug=0)
        b = _embedd(b, self.p, pa)
        if self.centered:
            return b, omegas
        else:
            raise NotImplementedError("Not implemented")
            # sub_means = self._sample_means[:, [j] + pa]
            # nus = _noise_means_from_B(B, sub_I, sub_means, self.n_obs)[:, 0]
            # return b, nus, omegas

    def _omegas_from_b(self, j, pa, b):
        """Given the regression coefficients for the jth variable, compute the
        variance of its noise terms.
        """
        # variable j has not received interventions: its noise-term
        # variance is constant across environments
        if j not in self.I:
            omega = self._pooled_covariance[j, j] - self._pooled_covariance[j, pa] @ b
            omegas = np.ones(self.e, dtype=float) * omega
        if j in self.I:
            I_B = -_embedd(b, self.p, pa)
            I_B[j] = 1
            omegas = I_B @ self._sample_covariances @ I_B
            # for e,cov in enumerate(sample_covariances):
            #     omegas[e] = cov[j,j] - cov[j,pa] @ b
            #     # Why does the above (commented) not work?  Idea: We're
            #     # not asking about the variance of the conditional
            #     # distribution of Xj given its parents in environment e,
            #     # as this would mean the regression coefficients would be
            #     # allowed to change within environments. We want to know
            #     # the variance of the residuals resulting from using the
            #     # regression coefficients in B.
        return omegas

    def _b_from_omegas(self, j, pa, omegas):
        """Regress j on its parents from the weighted covariance matrix,
        where the covariance matrix from each environment is weighted by
        the number of observation and the noise-term variance of j for
        that environment.
        """
        # TODO: Can further optimize -> if j is in I, the noise-term
        # variances (omegas) are the same across environments and thus
        # the weighted covariance is simply the pooled covariance
        # (computed in the class init)
        # if j in self.I:
        #     weighted_covariance = self._pooled_covariance
        # else:
        if j not in self.I:
            weighted_covariance = self._pooled_covariance
        else:
            weights = self.n_obs / omegas
            weights /= sum(weights)
            weighted_covariance = np.sum(self._sample_covariances * np.reshape(weights, (self.e, 1, 1)), axis=0)
        return _regress(j, pa, weighted_covariance)

    def _alternating_mle(self, j, pa, debug=0):
        # If j has no parents, can directly estimate omegas
        if len(pa) == 0:
            b = np.array([])
            omegas = self._omegas_from_b(j, pa, b)
            return b, omegas
        else:
            # Set starting point for optimization procedure
            b_0 = _regress(j, pa, self._pooled_covariance)
            omegas_0 = self._omegas_from_b(j, pa, b_0)

            # Components for the alternating (EM-like) optimization procedure
            # "Expectation" function: given b compute the noise-term variances
            def e_func(b):
                return self._omegas_from_b(j, pa, b)

            # "Maximization" function: given variances of noise terms, compute B
            def m_func(omegas):
                return self._b_from_omegas(j, pa, omegas)

            # Distance to check for convergence: max of L1 distance between
            # successive b's and successive omegas
            def dist(prev_b, prev_omegas, b, omegas):
                return max(abs(b - prev_b).max(), abs(omegas - prev_omegas).max())

            # Keep track of the objective function if debugging is desired
            if debug:
                def objective(B, omegas):
                    return log_likelihood.local(j, b, omegas, self._sample_covariances, self.n_obs)
            else:
                objective = None

            # Run procedure
            print("    Running alternating optimization procedure") if debug else None
            (b, omegas) = _em_like(e_func, m_func, b_0, omegas_0, dist,
                                   tol=self.tol, max_iter=self.max_iter, objective=objective, debug=debug)
            return b, omegas

# --------------------------------------------------------------------
# Functions for the alternating maximization procedure used to find
# the MLE


def _regress(j, pa, cov):
    """Compute the regression coefficients from the covariance matrix i.e.
    b = Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
    """
    return np.linalg.solve(cov[pa, :][:, pa], cov[j, pa])


def _embedd(b, p, idx):
    """Create a new vector with elements in idx corresponding to the
    elements in b and zeros elsewhere."""
    vector = np.zeros(p, dtype=b.dtype)
    vector[idx] = b
    return vector








def _noise_means_from_B(B, I, sample_means, n_obs):
    # TODO
    e, p = sample_means.shape
    I_B = (np.eye(p) - B)
    # This gives the noise term means for each environment (as if all
    # variables were intervened, on all environments)
    means = sample_means @ I_B
    # Compute the noise means for each variable across the
    # non-intervened environments
    scaled = sample_means * np.reshape(n_obs, (e, 1))
    for j in range(p):
        non_intervened = np.where([j not in i for i in I])[0]
        if len(non_intervened) > 0:
            # pooled is a 1xp array, containing the mean for each
            # variable, pooled across the "non_intervened" environments
            pooled = scaled[non_intervened].sum(axis=0) / n_obs[non_intervened].sum()
            means[non_intervened, j] = pooled @ I_B[:, j]
    return means

def _em_like(e_func, m_func, M_0, E_0, dist, tol=1e-6, assume_convex=False, max_iter=100, objective=None, debug=False):
    """
    EM-like alternating optimization procedure.

    Parameters
    ----------
    e_func : function
        "Expectation" function
    m_func : function
        "Maximization" function
    M_0 : any
        initial value for the expectation function
    E_0 : any
        initial value for the maximization function. In principle, not
        needed as the first step of the algorithm overrides this
        value; however, if the initial values satisfy convergence this
        allows us to return them : in a user-transparent way).
    dist : function
        The distance function used to check convergence. Takes as
        arguments the E,M from the previous and current iterations and
        returns a float
    tol : float
        Convergence tolerance. Stop if distance between consecutive
        iterations is below this threshold.
    assume_convex : boolean
        if the objective is assumed convex. If true, an increase in
        the distance of iterations is due to numerical issues, and we
        stop the iterations at this point, even if the tolerance has
        not been reached
    objective : function
        For debugging purposes, the objective function
    debug : boolean
        Print results of each iteration

    Returns
    -------
    M : any
        the approximated maximizer of the "Maximization quantity"
    E : any
        the approximated maximizer of the "Expectation quantity"

    """
    print("     ", end="") if debug else None
    prev_E = E = E_0
    prev_M = M = M_0
    prev_delta = np.inf
    for i in range(max_iter):
        E = e_func(M)  # E-step
        M = m_func(E)  # M-step
        delta = dist(prev_M, prev_E, M, E)
        # Debug outputs
        if objective is not None and debug:
            try:
                value = objective(M, E)
                print(" %0.16f (%0.16f)" % (delta, value), end="")
            except Exception as e:
                print(" %0.16f (%s)" % (delta, e), end="")
        elif debug:
            print(" %0.4f" % delta, end="")
            # Check stopping conditions
        if delta <= tol:
            print() if debug else None
            return M, E
        elif assume_convex and delta > prev_delta:
            # if program is convex, an increase in the distance
            # between iterations is due to numerical issues. Thus we
            # return the results from the previous iteration as we
            # assume that's as close as we can get
            print() if debug else None
            return prev_M, prev_E
        else:
            prev_E = E
            prev_M = M
            prev_delta = delta
    print(" MAX ITER REACHED") if debug else None
    return M, E


def ddof_full(A, I, e, centered=True):
    """Compute the number of free parameters in a model specified by the
    DAG adjacency and the intervention targets.

    Parameters
    ----------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.
    I : list of sets
        The sets of variables which have received interventions in
        each environment.
    centered : bool
        If we are also fitting an intercept or we assume the data to
        be centered.

    Returns
    -------
    ddof : int
        the number of free parameters in the model


    """
    p = len(A)
    # We estimate a weight for each edge
    n_edges = (A != 0).sum()
    n_variances = len(I) * e + (p - len(I))
    n_intercepts = 0 if centered else n_variances
    return n_edges + n_variances + n_intercepts


def ddof_local(j, pa, I, e, centered=True):
    """Compute the number of free parameters that are estimated for a
    variable in the model, given its parents and the intervention
    targets.

    Parameters
    ----------
    j : int
        The variable's index.
    pa : list(int)
        The set of parents.
    I : list of sets
        The sets of variables which have received interventions in
        each environment.
    centered : bool
        If we are also fitting an intercept or we assume the data to
        be centered.

    Returns
    -------
    ddof : int
        the number of free parameters

    """
    n_variances = e if j in I else 1
    n_intercepts = 0 if centered else n_variances
    return len(pa) + n_variances + n_intercepts
