"""
sdagd_classifier.py

An online logistic-regression classifier implementing a diagonal,
single-sample (stochastic) adaptation of Approximate Greatest Descent
(AGD) / Stochastic Diagonal AGD (SDAGD), following:

  B.S. Goh, "Greatest Descent Algorithms in Unconstrained Optimization,"
  J. Optim. Theory Appl. 142, 275-289 (2009).
  B.S. Goh, "Approximate Greatest Descent Methods for Optimization with
  Equality Constraints," J. Optim. Theory Appl. 148, 505-527 (2011).
  Harno et al., "Approximate greatest descent in neural network
  optimization" (SDAGD for MLP weight updates).

MECHANISM (two-phase switching, per Goh):
  Phase 1 (far from the minimum): the current local spherical search
  region of radius r does NOT contain the minimum point. The greatest
  descent direction is approximated by a Levenberg-Marquardt-like
  formula, with the damping term mu chosen so the step lands exactly on
  the boundary of the search region (||d|| = r). This is found by
  bisection on mu, since ||d(mu)|| is monotonically decreasing in mu.

  Phase 2 (near the minimum): once the unconstrained diagonal Newton
  step already lies inside the search region (||d_Newton|| <= r), the
  search region contains the minimum, so the iteration switches directly
  to the (diagonal-approximated) Newton step for fast local convergence.

  The search region radius r is then adapted every step using a
  trust-region-style ratio of actual to model-predicted loss decrease
  (grow r on good steps, shrink it on poor/rejected ones).

This is a single-sample (stochastic) diagonal version so it can be
called every trade result via partial_fit, matching the relay's online
learning loop -- Goh's original AGD is a batch/full-gradient method,
and SDAGD in the literature is used for backprop batches rather than
this single-sample per-trade setting, so this is an adaptation of the
published mechanism to that use case, not a verbatim reproduction.
"""

import numpy as np


class SDAGDClassifier:
    """
    Online binary classifier using diagonal SDAGD updates.
    Drop-in-compatible with the relay's usage pattern:
        clf = SDAGDClassifier(n_features)
        clf.partial_fit(X, y, classes=[0, 1])
        clf.predict_proba(X)
    """

    def __init__(self, n_features, r_init=1.0, r_min=1e-3, r_max=10.0,
                 hessian_decay=0.99, l2=1e-4, bisect_iters=25,
                 random_state=42):
        rng = np.random.RandomState(random_state)
        self.theta = rng.normal(0, 0.01, size=n_features + 1)  # +1 bias
        self.h_diag = np.ones(n_features + 1)
        self.r = r_init          # trust-region (spherical search) radius
        self.r_min = r_min
        self.r_max = r_max
        self.hessian_decay = hessian_decay
        self.l2 = l2
        self.bisect_iters = bisect_iters
        self.classes_ = np.array([0, 1])
        self.n_updates = 0
        self.last_phase = None   # "newton" or "boundary", for logging

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _augment(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _boundary_step(self, g, h):
        """
        Bisect for mu such that ||g / (h + mu)|| == r (spherical boundary).
        ||d(mu)||^2 is strictly decreasing in mu for mu >= 0, so bisection
        is well-posed. Returns the step d = -g / (h + mu).
        """
        mu_lo, mu_hi = 0.0, 1.0
        # expand mu_hi until the norm at mu_hi is below r (or hit a cap)
        for _ in range(50):
            d_hi = g / (h + mu_hi)
            if np.linalg.norm(d_hi) <= self.r:
                break
            mu_hi *= 2.0

        for _ in range(self.bisect_iters):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            d_mid = g / (h + mu_mid)
            if np.linalg.norm(d_mid) > self.r:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid

        mu = mu_hi
        return -g / (h + mu)

    def partial_fit(self, X, y, classes=None):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        x_aug_batch = self._augment(X)
        for x_aug, yi in zip(x_aug_batch, y):
            self._step(x_aug, float(yi))
            self.n_updates += 1
        return self

    def _step(self, x_aug, yi):
        p = self._sigmoid(np.dot(x_aug, self.theta))
        p_c = np.clip(p, 1e-9, 1 - 1e-9)
        loss_before = -(yi * np.log(p_c) + (1 - yi) * np.log(1 - p_c))

        g = (p - yi) * x_aug + self.l2 * self.theta
        h_sample = p * (1 - p) * (x_aug ** 2) + self.l2
        self.h_diag = self.hessian_decay * self.h_diag + (1 - self.hessian_decay) * h_sample
        h = self.h_diag

        # ── Phase check: does the search region contain the minimum? ──
        d_newton = -g / h
        if np.linalg.norm(d_newton) <= self.r:
            d = d_newton
            self.last_phase = "newton"
        else:
            d = self._boundary_step(g, h)
            self.last_phase = "boundary"

        theta_trial = self.theta + d
        p_trial = self._sigmoid(np.dot(x_aug, theta_trial))
        p_trial_c = np.clip(p_trial, 1e-9, 1 - 1e-9)
        loss_after = -(yi * np.log(p_trial_c) + (1 - yi) * np.log(1 - p_trial_c))

        actual_decrease = loss_before - loss_after
        # quadratic model's predicted decrease: -(g.d + 0.5 d^T H d)
        predicted_decrease = -(np.dot(g, d) + 0.5 * np.dot(h * d, d))
        predicted_decrease = max(predicted_decrease, 1e-12)  # guard div-by-0
        rho = actual_decrease / predicted_decrease

        # ── Trust-region radius update ─────────────────────────────
        if rho > 0.75:
            self.r = min(self.r_max, self.r * 2.0)
        elif rho < 0.25:
            self.r = max(self.r_min, self.r * 0.5)

        if actual_decrease > 0:
            self.theta = theta_trial
        # else: reject the step, keep old theta, r already shrunk above

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        x_aug = self._augment(X)
        p1 = self._sigmoid(x_aug @ self.theta)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
