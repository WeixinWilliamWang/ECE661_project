
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
# Local modules
from optim import SGLD
# Transformer backbone
from diffusionmodel import MLPDiffusionEps
from transformer import Transformer
from denoiser import OneDimCNN
from copy import deepcopy
import math
from bandit import safe_multivariate_normal
import wandb


class MovielensMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class SmallMLP(nn.Module):
    """Simple binary MLP: 784  -> 128 -> 1"""

    def __init__(self, in_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # output shape (batch,)




class DiffusionPriorMLP(object):
  """Diffusion Prior with pluggable ε-network architecture.

  Parameters
  ----------
  arch : str, default "mlp"
      Either "mlp" or "transformer".
  hidden : int
      Width of hidden layers (also transformer model width).
  n_layers : int
      For MLP: total FC layers; for Transformer: encoder layers.
  n_head : int, default 8
      Transformer attention heads (ignored for MLP).
  """

  def __init__(self, d, T, alpha=0.97, hidden=256, n_layers=6, n_head=8,
               device='cpu', schedule="const", arch="cnn1d", ema_decay=0.999,
               layer_channels=None, model_dim=256, kernel_size=7,
               proj_dim=None, proj_layers: int = 1):
    self.d = d
    self.T = T
    if schedule=="cosine":
      s = 0.008
      ts = np.linspace(0, self.T, self.T+1) / self.T
      f = lambda t: np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
      alpha_bar = f(ts) / f(0)
    elif schedule=="linear":
      # Linear beta schedule: beta_t linearly spaced from 1e-4 to 2e-2
      betas = np.linspace(1e-4, 2e-2, self.T)  # (T,)
      alpha_bar = np.empty(self.T + 1)
      alpha_bar[0] = 1.0
      alpha_bar[1:] = np.cumprod(1.0 - betas)
    else:
      alpha_bar = alpha ** np.arange(0, self.T+1)
    self.alpha_bar = alpha_bar
    self.alpha = np.empty(self.T + 1)
    self.alpha[0] = 1.0
    self.alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
    self.beta = 1 - self.alpha

    # beta_tilde per DDPM definition
    self.beta_tilde = np.zeros(self.T + 1)
    self.beta_tilde[1:] = (1 - self.alpha_bar[:-1]) * self.beta[1:] / (1 - self.alpha_bar[1:])

    self.device = device

    arch = arch.lower()
    if arch == "mlp":
      self.eps_net = MLPDiffusionEps(d_in=d, width=hidden, n_layers=n_layers).to(device)
    elif arch == "transformer":
      # ---- Derive parameter sizes from SmallMLP definition ----
      dummy_mlp = MovielensMLP(in_dim=20)
      state_dict = dummy_mlp.state_dict()
      layers = []
      layer_names = []
      for l in state_dict:
          shape = state_dict[l].shape
          layers.append(np.prod(shape))
          layer_names.append(l)

      # ---- Transformer hyper-parameters (can be externalised) ----
      trans_cfg = dict(
          n_embd           = 1024,
          n_layer          = 12,
          n_head           = 16,
          split_policy     = "layer_by_layer",
          use_global_residual = False,
          condition        = "no",
      )

      self.eps_net = Transformer(layers, layer_names, **trans_cfg).to(device)
    elif arch == "cnn1d":
      if layer_channels is None:
        layer_channels = [1, 64, 128, 256, 512, 256, 128, 64, 1]
      self.eps_net = OneDimCNN(layer_channels=layer_channels,
                               model_dim=model_dim,
                               kernel_size=kernel_size,
                               proj_dim=proj_dim,
                               proj_layers=proj_layers).to(device)
    else:
      raise ValueError(f"Unknown eps-net architecture '{arch}'")

  def train(self, S0, epochs=100, batch=512, lr=1e-3, weight_decay=1e-5,
            log_prefix=None):
    S0 = torch.tensor(S0, dtype=torch.float32, device=self.device)
    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(self.eps_net.parameters(), lr=lr, betas=(0.9, 0.9995), weight_decay=weight_decay)

    alpha_bar_tensor = torch.tensor(self.alpha_bar, dtype=torch.float32, device=self.device)  # (T+1,)
    self.eps_net.train()
    epoch_losses = []
    pbar = tqdm(range(epochs), desc="Epoch")
    step_idx = 0

    dataset = TensorDataset(S0)

    for epoch in pbar:
      loader = DataLoader(dataset, batch_size=batch, shuffle=True)
      epoch_loss = 0.0
      n_samples = 0
      for (x0,) in loader:
        # Uniformly sample timesteps for each sample in the batch
        t = torch.randint(1, self.T + 1, (x0.size(0),), device=self.device)
        eps = torch.randn_like(x0)
        alpha_bar_t = alpha_bar_tensor[t].unsqueeze(1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
        # Use raw timestep index t (no normalization) as in original G.pt implementation
        pred = self.eps_net(x_t, t.float())
        target = eps
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eps_net.parameters(), max_norm=1.0)
        optimizer.step()
        step_idx += 1
        # accumulate total loss weighted by batch size for exact epoch mean
        bs = x0.size(0)
        epoch_loss += loss.item() * bs
        n_samples += bs
      avg_loss = epoch_loss / max(n_samples, 1)
      pbar.set_postfix(loss=f"{avg_loss:.4f}")
      epoch_losses.append(avg_loss)
      if log_prefix is not None:
        wandb.log({f"{log_prefix}/epoch": epoch + 1,
                   f"{log_prefix}/loss": avg_loss})
      print(f"{log_prefix} loss: {avg_loss:.4f}")
    return np.array(epoch_losses)

  def conditional_prior_mean(self, S, t):
    # Algorithm 2 in Ho et al. (2020)
    # Denoising Diffusion Probabilistic Models
    with torch.no_grad():
      S_torch = torch.as_tensor(S, dtype=torch.float32, device=self.device)
      # Use raw timestep index t consistently with training
      t_vec = torch.full((S_torch.size(0),), float(t), dtype=torch.float32, device=self.device)
      pred = self.eps_net(S_torch, t_vec)
    epsilon = pred.detach().cpu().numpy()
    S0 = (S - np.sqrt(1 - self.alpha_bar[t]) * epsilon) / np.sqrt(self.alpha_bar[t])
    w0 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
    wt = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
    mu = w0 * S0 + wt * S
    return mu

    
  def sample(self, n):
    # reverse process sampling
    # S = np.zeros((self.T + 1, n, self.d))
    # S[self.T, :, :] = np.random.randn(n, self.d)
    # for t in range(self.T, 0, -1):
    #   mu = self.conditional_prior_mean(S[t, :, :], t)
    #   S[t - 1, :, :] = mu + np.sqrt(self.beta_tilde[t]) * np.random.randn(n, self.d)
    #   S[t - 1, :, :] = np.minimum(np.maximum(S[t - 1, :, :], -100), 100)  # numerical stability (just in case)

    # return S
    x = np.random.randn(n, self.d)
    for t in range(self.T, 0, -1):
      x = self.conditional_prior_mean(x, t) + np.sqrt(self.beta_tilde[t]) * np.random.randn(n, self.d)
      x = np.minimum(np.maximum(x, -100), 100)  # numerical stability (just in case)
    return [x]

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
      S[t - 1, :] = safe_multivariate_normal(mu_hat, Sigma_hat)
      S[t - 1, :] = np.minimum(np.maximum(S[t - 1, :], -100), 100)  # numerical stability (just in case)

    return S

  def posterior_sample_map(self, map_lambda):
    # reverse process sampling with evidence
    S = np.zeros((self.T + 1, self.d))
    for t in range(self.T + 1, 0, -1):
      # posterior distribution
      if t == self.T + 1:
        mu0 = np.zeros(self.d)
        den = max(self.alpha_bar[t - 1], 1e-12)
        Sigma0 = np.eye(self.d) / den
      else:
        mu = np.squeeze(self.conditional_prior_mean(S[[t], :], t))
        Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)  # zero covariance in stage 1 of the reverse process
        denom = max(self.alpha_bar[t - 1], 1e-12)
        mu0 = mu / np.sqrt(denom)
        Sigma0 = Sigma / denom

      mu_hat, Sigma_hat = map_lambda(mu0, Sigma0)
      mu_hat *= np.sqrt(self.alpha_bar[t - 1])
      Sigma_hat *= self.alpha_bar[t - 1]

      # posterior sampling (simple, without fallback)
      S[t - 1, :] = safe_multivariate_normal(mu_hat, Sigma_hat)
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
        S_torch = torch.tensor(S[[t], :], dtype=torch.float32, device=self.device)
        eps_pred = self.eps_net(S_torch, torch.tensor([t], dtype=torch.float32, device=self.device)/self.T)
        epsilon = eps_pred.detach().cpu().numpy()
        score = - epsilon / np.sqrt(1 - self.alpha_bar[t])
        s0 = (S[t, :] + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])
        w0 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
        wt = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
        mu = w0 * s0 + wt * S[t, :]
        Sigma = np.maximum(self.beta_tilde[t], 1e-6) * np.eye(self.d)  # zero covariance in stage 1 of the reverse process

      # posterior sampling
      S[t - 1, :] = safe_multivariate_normal(mu, Sigma) + loglik_grad(s0)
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
    # Matches MovielensMLP: ReLU activation, Linear(d_in, hidden) -> ReLU -> Linear(hidden,1)
    w1_end = hidden * d_in
    b1_end = w1_end + hidden
    w2_end = b1_end + hidden
    # Unpack
    w1 = theta_vec[:w1_end].view(hidden, d_in)
    b1 = theta_vec[w1_end:b1_end]
    w2 = theta_vec[b1_end:w2_end]
    b2 = theta_vec[w2_end]
    h = torch.relu(X_tensor.matmul(w1.t()) + b1)
    out = h.matmul(w2.unsqueeze(1)).squeeze(1) + b2
    return out

  # --------------------  Main SGLD routine --------------------
  def _sgld_appr(self, X, y, mu, Sigma, alpha_scaling,
                 num_steps=1, step_size=0.05,
                 link_func=None, sigma=1.0, noise_scale=0.01,
                 predict_func=None):
    """Approximate posterior sample via manual SGLD-style updates."""

    if predict_func is None and link_func is None:
      link_func = lambda z: z

    d = self.d
    if X.size == 0:
      return np.random.multivariate_normal(mu, Sigma)

    if isinstance(self.device, torch.device):
      device = self.device
    else:
      try:
        device = torch.device(self.device)
      except (TypeError, RuntimeError):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
      device = torch.device("cpu")
    dtype = torch.float32

    alpha_scale = float(np.sqrt(max(alpha_scaling, 1e-12)))
    Xt = torch.from_numpy(X / alpha_scale).to(device=device, dtype=dtype)
    yt = torch.from_numpy(y).to(device=device, dtype=dtype)
    mu_t = torch.from_numpy(mu).to(device=device, dtype=dtype)

    theta = mu_t.clone().detach()
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_inv_t = torch.from_numpy(Sigma_inv).to(device=device, dtype=dtype)
    if sigma !=0:
      inv_sigma2 = 1.0 / (sigma ** 2)
    else:
      inv_sigma2 = 1.0
    noise_std = noise_scale * math.sqrt(2.0 * step_size)

    for _ in range(max(1, num_steps)):
      theta.requires_grad_(True)

      if predict_func is None:
        z = Xt.matmul(theta)
        g_z = link_func(z) if link_func is not None else z
      else:
        g_z = predict_func(theta, Xt)

      resid = g_z - yt
      loss_data = inv_sigma2 * (resid ** 2).sum()
      prior_term = 0.5 * (theta - mu_t).matmul(Sigma_inv_t).dot(theta - mu_t)
      loss = loss_data + prior_term
      loss.backward()

      with torch.no_grad():
        grad = theta.grad
        if grad is None:
          break
        theta = theta - step_size * grad
        if noise_scale > 0:
          theta = theta + noise_std * torch.randn_like(theta)
        theta = torch.clamp(theta, -1e3, 1e3).detach()

    theta_np = theta.to("cpu").numpy()
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
    valid = ~np.isnan(yh)
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    # S = np.zeros((self.T + 1, d))

    # initialize prior parameters for ℓ = T (full covariance matrix)
    mu = np.zeros(d)
    Sigma = np.eye(d)

    for t in range(self.T + 1, 0, -1):
      alpha_scaling = self.alpha_bar[t-1]

      # SGLD K-step sample given current (mu, prior_var)
      # print(f"SGLD K-step sample given current (mu, prior_var) for t={t}")
      theta_prev = self._sgld_appr(
          X_obs, y_obs, mu, Sigma, alpha_scaling,
          num_steps=num_steps_sgld, step_size=step_size_sgld,
          link_func=link_func, sigma=sigma, noise_scale=noise_scale,
          predict_func=predict_func)

      x = np.clip(theta_prev, -100, 100)
      if t-1 != 0:
        # Update prior parameters for next iteration using new sample
        mu = self.conditional_prior_mean(x[None, :], t-1)[0]
        Sigma = float(max(self.beta_tilde[t-1], 1e-6)) * np.eye(d)
    return [x]



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
    valid = ~np.isnan(yh)
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    # S = np.zeros((self.T + 1, d))
    # θ_L ∼ N(0,I)
    # S[self.T, :] = np.random.randn(d)
    x = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score network → epsilon  -> score
      t_frac = torch.tensor([t / self.T], dtype=torch.float32, device=self.device)
      with torch.no_grad():
        x_torch = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        pred_torch = self.eps_net(x_torch, t_frac)
      pred = np.squeeze(pred_torch.detach().cpu().numpy())
      score = - pred / np.sqrt(1 - self.alpha_bar[t])

      # hat θ0(θ_t, t)  (Alg2 line6)
      theta0_hat = (x + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])

      # Prior drift (Alg2 line8) : μ_t & β_t
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      theta_prime = np.sqrt(self.alpha[t]) *(1-self.alpha_bar[t-1])* x / (1-self.alpha_bar[t])  + np.sqrt(self.alpha_bar[t-1]) * self.beta[t] * theta0_hat / (1-self.alpha_bar[t])  + z_t
      # Likelihood drift (line9) via one-step SGD on weighted squared loss

      if X_obs.size == 0:
        theta_prev = theta_prime
      else:
        device = self.device
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

      x = np.clip(theta_prev, -100, 100)

    return [x]

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
    valid = ~np.isnan(yh)
    X_obs = Xh[valid, :]
    y_obs = yh[valid]

    d = self.d
    # S = np.zeros((self.T + 1, d))
    # S[self.T, :] = np.random.randn(d)
    x = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score & θ̂0 (line4)
      t_frac = torch.tensor([t / self.T], dtype=torch.float32, device=self.device)
      with torch.no_grad():
        x_torch = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        pred_torch = self.eps_net(x_torch, t_frac)
      pred = np.squeeze(pred_torch.detach().cpu().numpy())
      score = - pred / np.sqrt(1 - self.alpha_bar[t])

      # hat θ0(θ_t, t)  (Alg2 line6)
      theta0_hat = (x + (1 - self.alpha_bar[t]) * score) / np.sqrt(self.alpha_bar[t])

      # Prior drift parameters (line6)
      c1 = np.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
      c2 = np.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t])
      m_t = c1 * x + c2 * theta0_hat

      # noise & radius (line5)
      z_t = np.sqrt(self.beta[t]) * np.random.randn(d)
      r_t = np.linalg.norm(z_t)

      theta_k = m_t + z_t  # θ_{ℓ-1,0}

      if X_obs.size != 0:
        # Prepare torch tensors
        device = self.device
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

      x = np.clip(theta_lm1_k, -100, 100)

    return [x]


  def posterior_sample_dps_acr(self, Xh, yh, Xt_cur, sigma=1.0, eta=0.05, h=1.0):
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
    valid = ~np.isnan(yh)
    X_hist = Xh[valid, :]
    y_hist = yh[valid]

    d = self.d
    S = np.zeros((self.T + 1, d))
    S[self.T, :] = np.random.randn(d)

    for t in range(self.T, 0, -1):
      # score & θ̂0 (line4)
      t_frac = torch.tensor([t / self.T], dtype=torch.float32, device=self.device)
      with torch.no_grad():
        S_torch = torch.as_tensor(S[[t], :], dtype=torch.float32, device=self.device)
        pred = np.squeeze(self.eps_net(S_torch, t_frac)).detach().cpu().numpy()
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
        pred = Xt.matmul(theta_t)
        loss = ((wt * (pred - yt) ** 2).sum()) / (sigma ** 2)
        loss.backward()
        optim.step()
        theta_t = theta_t.detach().cpu().numpy()
      S[t - 1, :] = np.clip(theta_t, -100, 100)

      return S


  # --------------------  Checkpoint helpers --------------------
  def state_dict(self):
    """Return state_dict of internal ε-network for serialization."""
    return self.eps_net.state_dict()

  def load_state_dict(self, state_dict):
    """Load parameters into internal ε-network (compatible with torch.load)."""
    self.eps_net.load_state_dict(state_dict)























