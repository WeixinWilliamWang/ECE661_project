import joblib
from joblib import Parallel, delayed
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from tqdm.auto import tqdm
import torch
from optim import SGLD
from optim import sgld  # functional SGLD update
import math
import torch.nn as nn


class KernelRegressor(object):
  def __init__(self, num_clusters=10, reg=1e-6):
    self.num_clusters = num_clusters
    self.reg = reg

  def fit(self, features, Y):
    kmeans = KMeans(self.num_clusters)
    self.centers = kmeans.fit(features).cluster_centers_
    dX2 = np.square(self.centers).sum(axis=-1)
    dXY = np.einsum("ik,jk->ij", self.centers, self.centers)
    d2 = dX2[:, np.newaxis] - 2 * dXY + dX2[np.newaxis, :]
    self.kernel_width = np.sqrt(np.sort(d2, axis=-1)[:, 1].mean())

    dX2 = np.square(features).sum(axis=-1)
    self.dY2 = np.square(self.centers).sum(axis=-1)
    dXY = np.einsum("ik,jk->ij", features, self.centers)
    d2 = dX2[:, np.newaxis] - 2 * dXY + self.dY2[np.newaxis, :]
    d2 -= d2.min(axis=-1)[:, np.newaxis]

    X = np.exp(- d2 / (2 * np.square(self.kernel_width)))
    X /= X.sum(axis=-1)[:, np.newaxis]
    self.Theta = np.linalg.inv(X.T.dot(X) + self.reg * np.eye(self.num_clusters)).dot(X.T.dot(Y))

  def predict(self, features):
    dX2 = np.square(features).sum(axis=-1)
    dXY = np.einsum("ik,jk->ij", features, self.centers)
    d2 = dX2[:, np.newaxis] - 2 * dXY + self.dY2[np.newaxis, :]
    d2 -= d2.min(axis=-1)[:, np.newaxis]

    X = np.exp(- d2 / (2 * np.square(self.kernel_width)))
    X /= X.sum(axis=-1)[:, np.newaxis]
    Y = X.dot(self.Theta)
    return Y


class DiffusionPrior(object):
  def __init__(self, d, T, alpha, reg=1e-6, tol=1e-4, hidden_size=None):
    self.d = d
    self.T = T
    self.alpha = alpha * np.ones(self.T + 1)
    self.alpha[0] = 1.0

    self.beta = 1 - self.alpha
    self.alpha_bar = np.cumprod(self.alpha)
    self.beta_tilde = np.zeros(self.T + 1)
    self.beta_tilde[1 :] = (1 - self.alpha_bar[: self.T]) * self.beta[1 :] / (1 - self.alpha_bar[1 :])

    # reverse process parameterization
    self.reg = reg  # least-squares regularization
    self.tol = tol  # fitting stopping tolerance
    if hidden_size is None:
      self.hidden_size = 10 * self.d
    else:
      self.hidden_size = hidden_size

  def train_stage(self, t, St, epsilon):
    # Equation 12 in Ho et al. (2020)
    # Denoising Diffusion Probabilistic Models
    regressor = MLPRegressor(hidden_layer_sizes=(self.hidden_size, self.hidden_size),
      alpha=self.reg, early_stopping=False, verbose=False, tol=self.tol, max_iter=1000)
    regressor.fit(St, epsilon)
    error = np.sqrt(np.square(regressor.predict(St) - epsilon).sum(axis=-1).mean())
    return regressor, error

  def train(self, S0):
    n = S0.shape[0]

    # diffusion using the forward process
    epsilon = np.random.randn(self.T + 1, n, self.d)
    S = np.zeros((self.T + 1, n, self.d))
    S[0, :, :] = S0
    for t in range(1, self.T + 1):
      S[t, :, :] = np.sqrt(self.alpha_bar[t]) * S0 + np.sqrt(1 - self.alpha_bar[t]) * epsilon[t, :, :]

    # reverse process learning
    output = Parallel(n_jobs=-1)(
      delayed(self.train_stage)(t, S[t, :, :], epsilon[t, :, :]) for t in tqdm(range(1, self.T + 1)))

    self.regressors = []
    self.regressors.append(None)
    errors = np.zeros(self.T)
    for t in range(self.T):
      self.regressors.append(output[t][0])
      errors[t] = output[t][1]

    return errors

  def no_train(self, S0):
    self.regressors = [None]
    for _ in range(self.T):
      self.regressors.append(TorchMLPRegressor(self.d, self.hidden_size))
    return self.regressors

  def conditional_prior_mean(self, S, t):
    # Algorithm 2 in Ho et al. (2020)
    # Denoising Diffusion Probabilistic Models
    epsilon = self.regressors[t].predict(S)
    S0 = (S - np.sqrt(1 - self.alpha_bar[t]) * epsilon) / np.sqrt(self.alpha_bar[t])
    w0 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
    wt = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
    mu = w0 * S0 + wt * S
    return mu

  def sample(self, n):
    # reverse process sampling
    S = np.zeros((self.T + 1, n, self.d))
    S[self.T, :, :] = np.random.randn(n, self.d)
    for t in range(self.T, 0, -1):
      mu = self.conditional_prior_mean(S[t, :, :], t)
      S[t - 1, :, :] = mu + np.sqrt(self.beta_tilde[t]) * np.random.randn(n, self.d)
      S[t - 1, :, :] = np.minimum(np.maximum(S[t - 1, :, :], -100), 100)  # numerical stability (just in case)

    return S

  def posterior_sample(self, theta_bar, Sigma_bar):
    # reverse process sampling with evidence
    S = np.zeros((self.T + 1, self.d))
    for t in range(self.T + 1, 0, -1):
      # diffused evidence
      theta_diff = np.sqrt(self.alpha_bar[t - 1]) * theta_bar
      Sigma_diff = self.alpha_bar[t - 1] * Sigma_bar
      Lambda_diff = np.linalg.inv(Sigma_diff)

      # posterior distribution
      if t == self.T + 1:
        Sigma_hat = np.linalg.inv(np.eye(self.d) + Lambda_diff)
        mu_hat = Sigma_hat.dot(Lambda_diff.dot(theta_diff))
      else:
        mu = np.squeeze(self.conditional_prior_mean(S[[t], :], t))
        Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)  # zero covariance in stage 1 of the reverse process
        Lambda = np.linalg.inv(Sigma)
        Sigma_hat = np.linalg.inv(Lambda + Lambda_diff)
        mu_hat = Sigma_hat.dot(Lambda.dot(mu) + Lambda_diff.dot(theta_diff))

      # posterior sampling
      S[t - 1, :] = np.random.multivariate_normal(mu_hat, Sigma_hat)
      S[t - 1, :] = np.minimum(np.maximum(S[t - 1, :], -100), 100)  # numerical stability (just in case)

    return S

  def posterior_sample_map(self, map_lambda):
    # reverse process sampling with evidence
    S = np.zeros((self.T + 1, self.d))
    for t in range(self.T + 1, 0, -1):
      # posterior distribution
      if t == self.T + 1:
        mu0 = np.zeros(self.d)
        Sigma0 = np.eye(self.d) / self.alpha_bar[t - 1]
      else:
        mu = np.squeeze(self.conditional_prior_mean(S[[t], :], t))
        Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)  # zero covariance in stage 1 of the reverse process
        mu0 = mu / np.sqrt(self.alpha_bar[t - 1])
        Sigma0 = Sigma / self.alpha_bar[t - 1]

      mu_hat, Sigma_hat = map_lambda(mu0, Sigma0)
      mu_hat *= np.sqrt(self.alpha_bar[t - 1])
      Sigma_hat *= self.alpha_bar[t - 1]

      # posterior sampling
      S[t - 1, :] = np.random.multivariate_normal(mu_hat, Sigma_hat)
      S[t - 1, :] = np.minimum(np.maximum(S[t - 1, :], -100), 100)  # numerical stability (just in case)

    return S

  def posterior_sample_grad(self, loglik_grad):
    # reverse process sampling with evidence
    S = np.zeros((self.T + 1, self.d))
    for t in range(self.T + 1, 0, -1):
      # posterior distribution
      if t == self.T + 1:
        s0 = np.zeros(self.d)
        mu = np.zeros(self.d)
        Sigma = np.eye(self.d)
      else:
        # epsilon to score conversion based on (29) in Chung et al. (2023)
        # Diffusion Posterior Sampling for General Noisy Inverse Problems
        epsilon = np.squeeze(self.regressors[t].predict(S[[t], :]))
        score = - epsilon / np.sqrt(1 - self.alpha_bar[t])
        s0 = (S[t, :] + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])
        w0 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
        wt = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
        mu = w0 * s0 + wt * S[t, :]
        Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)  # zero covariance in stage 1 of the reverse process

      # posterior sampling
      S[t - 1, :] = np.random.multivariate_normal(mu, Sigma) + loglik_grad(s0)
      S[t - 1, :] = np.minimum(np.maximum(S[t - 1, :], -100), 100)  # numerical stability (just in case)

    return S


  # --------------------  Algorithm-5  (one-step SGLD) --------------------
  # --------------------  Helper: MLP forward --------------------
  @staticmethod
  def _mlp_forward_torch(theta_vec, X_tensor, d_in, hidden):
    """Forward pass of a fixed two-layer tanh MLP with flattened params.

    Parameter layout must match Synthetic.py::SmallMLP flattening:
      [w1 (hidden*d), b1 (hidden), w2 (hidden), b2 (1)]
    """
    w1_end = hidden * d_in
    b1_end = w1_end + hidden
    w2_end = b1_end + hidden
    # Unpack
    w1 = theta_vec[:w1_end].view(hidden, d_in)
    b1 = theta_vec[w1_end:b1_end]
    w2 = theta_vec[b1_end:w2_end]
    b2 = theta_vec[w2_end]
    h = torch.tanh(X_tensor.matmul(w1.t()) + b1)
    out = h.matmul(w2.unsqueeze(1)).squeeze(1) + b2
    return out

  # --------------------  Main SGLD routine --------------------
  def _sgld_appr(self, X, y, mu, Sigma, alpha_scaling,
                 num_steps=1, step_size=0.05,
                 link_func=None, sigma=1.0, noise_scale=0.01,
                 predict_func=None):
    # ----- Likelihood model selection -----
    # If a custom `predict_func(θ, X_tensor)` is provided, it overrides linear model.
    if predict_func is None:
      # Plain linear bandit ⇒ allow optional link_func (identity by default)
      if link_func is None:
        link_func = lambda z: z

    d = self.d
    if X.size == 0:
      return np.random.multivariate_normal(mu, Sigma)

    Sigma_inv = np.linalg.inv(Sigma)
    # Remove manual noise coefficient; handled by SGLD optimizer
    device = 'cpu'
    Xt = torch.from_numpy(X / np.sqrt(alpha_scaling)).double().to(device)
    yt = torch.from_numpy(y).double().to(device)
    mu_t = torch.from_numpy(mu).double().to(device)
    Sigma_inv_t = torch.from_numpy(Sigma_inv).double().to(device)
    theta = mu_t.clone().detach().requires_grad_(True)
    inv_sigma2 = 1.0 / (sigma ** 2)
    # According to the performance, keep this design
    optimizer = SGLD([theta], lr=step_size, weight_decay=0.0, noise_scale=noise_scale)
    for _ in range(num_steps):
      optimizer.zero_grad()
      # Forward pass
      if predict_func is None:
        z = Xt.matmul(theta)
        g_z = link_func(z)
      else:
        # custom non-linear predictions operate on raw design matrix
        g_z = predict_func(theta, Xt)
      resid = g_z - yt
      loss_data = inv_sigma2 * (resid ** 2).sum()  # mean for scale invariance
      prior_term = 0.5 * (theta - mu_t).matmul(Sigma_inv_t).dot(theta - mu_t)
      loss = loss_data + prior_term
      loss.backward()
      # Gradient clipping to avoid exploding updates
      torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
      optimizer.step()
      # Clamp parameters to a reasonable range to avoid Inf
      theta.data.clamp_(-1e3, 1e3)
    theta_np = theta.detach().cpu().numpy()
    # guard against NaNs/Infs that may arise from numerical issues
    if not np.isfinite(theta_np).all():
      theta_np = np.nan_to_num(theta_np, nan=0.0, posinf=1e6, neginf=-1e6)
    return theta_np

  # # -------- Algorithm-4 : full posterior sampling via diffusion ----------
  # def posterior_sample_sgld(self, Xh, yh, link_func=None, link_grad=None, sigma=1.0,
  #                               num_steps_sgld=1, step_size_sgld=0.05, noise_scale=0.01):
  #   S = np.zeros((self.T + 1, self.d))

  #   # select observed entries (ignore zero rows)
  #   valid = yh != 0
  #   X_obs = Xh[valid, :]
  #   y_obs = yh[valid]

  #   # initial diffused sample at step L (T)
  #   mu = np.zeros(self.d)
  #   Sigma = np.eye(self.d)
  #   theta_L = self._sgld_appr(
  #       X_obs, y_obs, mu, Sigma,
  #       self.alpha_bar[self.T],
  #       num_steps=num_steps_sgld, step_size=step_size_sgld,
  #       link_func=link_func, sigma=sigma, noise_scale=noise_scale)
  #   S[self.T, :] = theta_L

  #   # iterate ℓ = T … 1
  #   for t in range(self.T, 0, -1):
  #     mu = np.squeeze(self.conditional_prior_mean(S[[t], :], t))
  #     Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)
  #     alpha_scaling = self.alpha_bar[t - 1]
  #     theta_t = self._sgld_appr(
  #         X_obs, y_obs, mu, Sigma, alpha_scaling,
  #         num_steps=num_steps_sgld, step_size=step_size_sgld,
  #         link_func=link_func, sigma=sigma, noise_scale=noise_scale)

  #     # sample from gaussian distribution
  #     S[t - 1, :] = np.random.multivariate_normal(theta_t, Sigma)

  #     S[t - 1, :] = np.minimum(np.maximum(S[t - 1, :], -100), 100)

  #   return S


  def posterior_sample_dpts(self, Xh, yh, link_func=None, sigma=1.0,
                           num_steps_sgld=5, step_size_sgld=0.05, noise_scale=0.01,
                           predict_func=None):
    """Diffusion Prior Thompson Sampling (Algorithm-1) via inner SGLD loops."""

    # observed data
    valid = yh != 0
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    S = np.zeros((self.T + 1, d))

    # initialize prior parameters for ℓ = T
    mu = np.zeros(d)
    Sigma = np.eye(d)

    for t in range(self.T + 1, 0, -1):
      alpha_scaling = self.alpha_bar[t-1]

      # SGLD K-step sample given current (mu,Sigma)
      theta_prev = self._sgld_appr(
          X_obs, y_obs, mu, Sigma, alpha_scaling,
          num_steps=num_steps_sgld, step_size=step_size_sgld,
          link_func=link_func, sigma=sigma, noise_scale=noise_scale,
          predict_func=predict_func)

      S[t-1, :] = np.clip(theta_prev, -100, 100)
      if t-1 != 0:
        # Update prior parameters for next iteration using new sample
        mu = np.squeeze(self.conditional_prior_mean(S[[t-1], :], t-1))
        Sigma = np.maximum(self.beta_tilde[t-1], 1e-6) * np.eye(d)
    return S



  def posterior_sample_dps(self, Xh, yh, sigma=1.0, eta=0.05, predict_func=None):
    """Diffusion Posterior Sampling (Algorithm-2).

    Parameters
    ----------
    Xh, yh : ndarray
        History design matrix and rewards (zeros on unused rows).
    link_grad : callable(theta, X, y) -> grad, optional
        Likelihood gradient evaluated at \hat θ_0
        Defaults to linear-Gaussian model.
    eta : float
        Likelihood-drift step size.
    """

    # observed data
    valid = yh != 0
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    S = np.zeros((self.T + 1, d))
    # θ_L ∼ N(0,I)
    S[self.T, :] = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score network → epsilon  -> score

      pred = np.squeeze(self.regressors[t].predict(S[[t], :]))
      score = - pred / np.sqrt(1 - self.alpha_bar[t])

      # hat θ0(θ_t, t)  (Alg2 line6)
      theta0_hat = (S[t, :] + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])

      # Prior drift (Alg2 line8) : μ_t & β_t
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      theta_prime = np.sqrt(self.alpha[t]) *(1-self.alpha_bar[t-1])* S[t, :] / (1-self.alpha_bar[t])  + np.sqrt(self.alpha_bar[t-1]) * self.beta[t] * theta0_hat / (1-self.alpha_bar[t])  + z_t
      # Likelihood drift (line9) via one-step SGD on weighted squared loss

      if X_obs.size == 0:
        theta_prev = theta_prime
      else:
        device = 'cpu'
        theta_prime = torch.from_numpy(theta_prime).double().to(device)
        Xt = torch.from_numpy(X_obs).double().to(device)
        yt = torch.from_numpy(y_obs).double().to(device)

        # ---------- manual gradient descent (matches eq. in paper) ----------
        theta_hat = torch.tensor(theta0_hat, dtype=torch.double, device=device, requires_grad=True)

        if predict_func is None:
          pred_y = Xt.matmul(theta_hat)
        else:
          pred_y = predict_func(theta_hat, Xt)

        resid = pred_y - yt
        loss = (resid ** 2).sum() / (sigma ** 2)
        loss.backward()

        with torch.no_grad():
          theta_updated = theta_prime - eta * theta_hat.grad
        theta_hat.grad.zero_()

        theta_prev = theta_updated.detach().cpu().numpy()

      S[t - 1, :] = np.clip(theta_prev, -100, 100)

    return S

  # # (Retained) helper gradient function may still be used elsewhere
  # def _linear_gaussian_grad_torch(self, theta_np, X_np, y_np, sigma):
  #   if X_np.size == 0:
  #     return np.zeros_like(theta_np)
  #   device = 'cpu'
  #   theta = torch.tensor(theta_np, dtype=torch.double, device=device, requires_grad=True)
  #   X = torch.tensor(X_np, dtype=torch.double, device=device)
  #   y = torch.tensor(y_np, dtype=torch.double, device=device)
  #   resid = X.matmul(theta) - y
  #   loss = (resid ** 2).sum() / (sigma ** 2)
  #   loss.backward()
  #   return theta.grad.detach().cpu().numpy()



  def posterior_sample_dmap(self, Xh, yh, link_grad=None, sigma=1.0,
                           K_inner=10, eta=0.01, predict_func=None):
    """Diffusion MAP estimation (Algorithm-3).

    Parameters
    ----------
    Xh, yh : ndarray
      History design matrix and rewards.
    link_grad : callable, optional
      Gradient of loss wrt θ. Defaults to linear-Gaussian.
    K_inner : int
      Number of gradient steps per diffusion layer.
    eta : float
      Base learning rate. Per-layer step uses same value.
    """

    # if link_grad is None:
    #   link_grad = lambda th, X, y: self._linear_gaussian_grad_torch(th, X, y, sigma)

    # observed data
    valid = yh != 0
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    S = np.zeros((self.T + 1, d))
    S[self.T, :] = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score & θ̂0 (line4)
      pred = np.squeeze(self.regressors[t].predict(S[[t], :]))
      score = - pred / np.sqrt(1 - self.alpha_bar[t])

      # hat θ0(θ_t, t)  (Alg2 line6)
      theta0_hat = (S[t, :] + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])

      # Prior drift parameters (line6)
      c1 = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
      c2 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
      m_t = c1 * S[t, :] + c2 * theta0_hat

      # noise & radius (line5)
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      r_t = np.linalg.norm(z_t)

      theta_k = m_t + z_t  # θ_{ℓ-1,0}

      if X_obs.size != 0:
        # Prepare torch tensors
        device = 'cpu'
        Xt = torch.from_numpy(X_obs).double().to(device)
        yt = torch.from_numpy(y_obs).double().to(device)
        theta_k = torch.from_numpy(theta_k).double().to(device)
        theta_hat = torch.tensor(theta0_hat, dtype=torch.double, device=device, requires_grad=True)
        m_t_torch = torch.from_numpy(m_t).double().to(device)

        for _ in range(K_inner):
          if predict_func is None:
            pred = Xt.matmul(theta_hat)
          else:
            pred = predict_func(theta_hat, Xt)

          resid = pred - yt
          loss = (resid ** 2).sum() / (sigma ** 2)
          loss.backward()

          with torch.no_grad():
            # gradient step
            theta_k.add_(theta_hat.grad, alpha=-eta)

            # project back to ball of radius r_t around m_t (if outside)
            diff = theta_k - m_t_torch
            diff_norm = diff.norm()
            if diff_norm > r_t:
              theta_k.copy_(m_t_torch + diff * (r_t / diff_norm))

          theta_hat.grad.zero_()

        theta_lm1_k = theta_k.detach().cpu().numpy()
      else:
        theta_lm1_k = theta_k  # no data, keep prior-drift sample

      S[t - 1, :] = np.clip(theta_lm1_k, -100, 100)

    return S

  def posterior_sample_dmap_new(self, Xh, yh, link_grad=None, sigma=1.0,
                                K_inner=10, eta=0.01, predict_func=None):
    """
    Diffusion MAP estimation (Algorithm-4: DPSG-MP [new]).

    Use autograd.grad to explicitly compute gradients, avoiding the issue
    where non-leaf tensors have .grad = None.
    """
    # 1. Filter valid observed data
    valid = yh != 0
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d

    # Initialize diffusion trajectory array S
    S = np.zeros((self.T + 1, d))
    # Initialize terminal noise (Alg4 Line 4)
    S[self.T, :] = np.random.randn(d)

    # Select PyTorch device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Preload observation tensors
    if X_obs.size != 0:
      Xt = torch.from_numpy(X_obs).double().to(device)
      yt = torch.from_numpy(y_obs).double().to(device)

    # Reverse diffusion loop t = T ... 1
    for t in range(self.T, 0, -1):
      # =========================================================
      # Stage 1: Unconditional transition (Alg4 Line 6–9)
      # =========================================================

      theta_l = S[[t], :]  # shape (1, d)

      # 1. Predict score
      eps_pred = np.squeeze(self.regressors[t].predict(theta_l))
      score = - eps_pred / np.sqrt(1 - self.alpha_bar[t])

      # 2. Tweedie estimate hat_theta_0
      theta0_hat_l = (theta_l + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])

      # 3. Compute mean m_t
      sqrt_alpha_l = np.sqrt(self.alpha[t])
      one_minus_alpha_bar_prev = 1 - self.alpha_bar[t - 1]
      one_minus_alpha_bar_t = 1 - self.alpha_bar[t]

      c1 = sqrt_alpha_l * one_minus_alpha_bar_prev / one_minus_alpha_bar_t
      c2 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / one_minus_alpha_bar_t

      m_t = c1 * theta_l + c2 * theta0_hat_l

      # 4. Sample noise and compute radius r_t
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      r_t = np.linalg.norm(z_t)

      # 5. Initial guess theta_{t−1,0}
      theta_lm1_k = m_t + z_t  # shape (1, d)

      # =========================================================
      # Stage 2: Inner loop with dynamic guidance (Alg4 Line 10–14)
      # =========================================================
      if X_obs.size != 0:
        # Convert to torch tensor for iterative updates on a sphere
        theta_k_torch = torch.from_numpy(theta_lm1_k).double().to(device).squeeze()
        m_t_torch = torch.from_numpy(m_t).double().to(device).squeeze()

        for _ in range(K_inner):
          # --- A. Compute dynamic hat_theta_0 (Alg4 Line 11) ---
          # Convert back to numpy for score network prediction
          theta_curr_np = theta_k_torch.detach().cpu().numpy().reshape(1, -1)

          if t > 1:
            # Use score network at time t−1
            eps_pred_inner = np.squeeze(self.regressors[t - 1].predict(theta_curr_np))
            score_inner = - eps_pred_inner / np.sqrt(1 - self.alpha_bar[t - 1])

            # Tweedie update for hat_theta_0
            theta0_hat_val = (theta_curr_np +
                              (1 - self.alpha_bar[t - 1]) * score_inner) / np.sqrt(self.alpha_bar[t - 1])
          else:
            # Final step (t = 1): hat_theta_0 = current value
            theta0_hat_val = theta_curr_np

          # --- B. Construct differentiable tensor for gradient computation ---
          theta_hat_torch = torch.from_numpy(theta0_hat_val).double().to(device).squeeze()
          theta_hat_torch.requires_grad_(True)

          # --- C. Forward pass and explicit gradient computation ---
          if predict_func is None:
            pred = Xt.matmul(theta_hat_torch)     # prediction from linear model
          else:
            pred = predict_func(theta_hat_torch, Xt)

          resid = pred - yt
          loss = (resid ** 2).sum() / (sigma ** 2)

          # Compute gradient explicitly (no .backward(), no .grad access)
          grad_val, = torch.autograd.grad(loss, theta_hat_torch)

          # --- D. Gradient update on the sphere ---
          with torch.no_grad():
            theta_k_torch.add_(grad_val, alpha=-eta)

            # Project back onto sphere centered at m_t with radius r_t
            diff = theta_k_torch - m_t_torch
            diff_norm = diff.norm()

            if diff_norm > r_t:
              theta_k_torch.copy_(m_t_torch + diff * (r_t / diff_norm))

        # Sync tensor update back to numpy
        theta_lm1_k = theta_k_torch.detach().cpu().numpy().reshape(1, -1)

      # Save the result of this reverse step
      S[t - 1, :] = np.clip(theta_lm1_k, -100, 100)

    return S



  def posterior_sample_dps_acr(self, Xh, yh, Xt_cur, sigma=1.0, eta=0.05, h=1.0, predict_func=None):
    """Diffusion Posterior Sampling with Action-Conditioned Reweighting.

    Parameters
    ----------
    Xh, yh : ndarray (history up to t-1)
    Xt_cur : ndarray (K , d)
        Context vectors of current arms at round t.
    sigma : float
        Observation noise std.
    eta : float
        Likelihood drift step size.
    h : float
        RBF kernel bandwidth.
    """

    def kernel(x, x0):
      return np.exp(- np.linalg.norm(x - x0) ** 2 / (2 * h ** 2))

    # pick observed data rows
    valid = yh != 0
    X_hist = Xh[valid, :]
    y_hist = yh[valid]

    d = self.d
    S = np.zeros((self.T + 1, d))
    S[self.T, :] = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score & θ̂0 (line4)
      pred = np.squeeze(self.regressors[t].predict(S[[t], :]))
      score = - pred / np.sqrt(1 - self.alpha_bar[t])

      # hat θ0(θ_t, t)  (Alg2 line6)
      theta0_hat = (S[t, :] + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])


      # provisional best arm using linear reward f(θ;x)=x·θ
      if X_hist.size != 0:
        mu_arms = X_hist.dot(theta0_hat)
        a_star_idx = np.argmax(mu_arms)
        x_star = X_hist[a_star_idx]

        w = np.apply_along_axis(lambda row: kernel(row, x_star), 1, X_hist)
        w /= w.sum()

      # ---------------- Prior drift ----------------
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      theta_prime = np.sqrt(self.alpha[t]) *(1-self.alpha_bar[t-1])/ (1-self.alpha_bar[t]) * S[t, :] + np.sqrt(self.alpha_bar[t-1]) * self.beta[t] / (1-self.alpha_bar[t]) * theta0_hat + z_t

      # ---------------- Likelihood drift via SGD ----------------
      if X_hist.size == 0:
        theta_t = theta_prime
      else:
        device = 'cpu'
        Xt = torch.from_numpy(X_hist).double().to(device)
        yt = torch.from_numpy(y_hist).double().to(device)
        wt = torch.from_numpy(w).double().to(device)

        theta_t = torch.tensor(theta_prime, dtype=torch.double, device=device, requires_grad=True)
        optim = torch.optim.SGD([theta_t], lr=eta)

        optim.zero_grad()
        if predict_func is None:
          pred = Xt.matmul(theta_t)
        else:
          pred = predict_func(theta_t, Xt)
        loss = ((wt * (pred - yt) ** 2).sum()) / (sigma ** 2)
        loss.backward()
        optim.step()
        theta_t = theta_t.detach().cpu().numpy()
      S[t - 1, :] = np.clip(theta_t, -100, 100)

    return S





















class TorchMLPRegressor:
  """Simple sklearn-like MLP regressor implemented with PyTorch.
  Supports fit(X, y) and predict(X).  If fit is never called, the network
  keeps random weights (useful for no_train mode)."""
  def __init__(self, d_in: int, hidden: int, lr: float = 1e-3, epochs: int = 200, device: str = 'cpu'):
    self.net = nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, d_in)
    ).double().to(device)
    self.lr = lr
    self.epochs = epochs
    self.device = device
    self._is_fitted = False

  def fit(self, X: np.ndarray, y: np.ndarray):
    X_t = torch.from_numpy(X).double().to(self.device)
    y_t = torch.from_numpy(y).double().to(self.device)
    optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    loss_fn = nn.MSELoss()
    self.net.train()
    for _ in range(self.epochs):
      optim.zero_grad()
      out = self.net(X_t)
      loss = loss_fn(out, y_t)
      loss.backward()
      optim.step()
    self._is_fitted = True
    return self

  def predict(self, X: np.ndarray) -> np.ndarray:
    self.net.eval()
    with torch.no_grad():
      X_t = torch.from_numpy(X).double().to(self.device)
      out = self.net(X_t).cpu().numpy()
    return out




















