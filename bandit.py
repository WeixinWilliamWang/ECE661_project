# ------------------ standard libs ------------------
import joblib
from joblib import Parallel, delayed
import numpy as np
import time
# Neural helpers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def posify_mtx(Sigma: np.ndarray, tol: float=1e-9):
  Sigma = 0.5 * (Sigma + Sigma.T) + 1e-6 * np.eye(Sigma.shape[0])
  return Sigma


def sigmoid(x):
  x = np.minimum(np.maximum(x, -10), 10)
  y = 1 / (1 + np.exp(- x))
  return y


def irls(X, y, theta0, Lambda0, theta, irls_error=1e-3, irls_num_iter=30):
  # iterative reweighted least squares for Bayesian logistic regression
  # Sections 4.3.3 and 4.5.1 in Bishop (2006)
  # Pattern Recognition and Machine Learning

  num_iter = 0
  while num_iter < irls_num_iter:
    theta_old = np.copy(theta)

    if y.size > 0:
      Xtheta = X.dot(theta)
      R = sigmoid(Xtheta) * (1 - sigmoid(Xtheta))
      Hessian = (X * R[:, np.newaxis]).T.dot(X) + Lambda0
      grad = (sigmoid(Xtheta) - y).dot(X) + Lambda0.dot(theta - theta0)
    else:
      R = np.zeros(0)
      Hessian = Lambda0
      grad = Lambda0.dot(theta - theta0)
    theta = np.linalg.solve(Hessian, Hessian.dot(theta) - grad)

    if np.linalg.norm(theta - theta_old) < irls_error:
      break;
    num_iter += 1

  converged = (num_iter < irls_num_iter)
  return theta, Hessian, converged


# Bandit environments and simulator

class GLMBandit(object):
  """GLM bandit."""

  def __init__(self, X, K, theta, mean_function="logistic", sigma=0.5):
    self.X_all = np.copy(X)  # matrix of all arm features
    self.K = K  # number of arms per round
    self.d = self.X_all.shape[1]  # number of features

    self.theta = np.copy(theta)  # model parameter
    self.mean_function = mean_function
    self.sigma = sigma  # reward noise

    self.randomize()

  def randomize(self):
    arms = np.random.choice(self.X_all.shape[0], self.K, replace=False)  # random subset of K arms
    self.X = self.X_all[arms, :]  # K x d matrix of arm features

    # mean rewards of all arms
    if self.mean_function == "linear":
      self.mu = self.X.dot(self.theta)
    elif self.mean_function == "logistic":
      self.mu = sigmoid(self.X.dot(self.theta))
    else:
      raise Exception("Unknown mean function.")

    self.best_arm = np.argmax(self.mu)  # optimal arm

    # generate random rewards
    if self.mean_function == "linear":
      self.rt = self.mu + self.sigma * np.random.randn(self.K)
    elif self.mean_function == "logistic":
      self.rt = (np.random.rand(self.K) < self.mu).astype(float)
    else:
      raise Exception("Unknown mean function.")

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "GLM bandit: %d dimensions, %d arms, %s mean function" % (self.d, self.K, self.mean_function)


class LinBandit(object):
  """Linear bandit."""

  def __init__(self, X, theta, sigma=0.5, labels=None, features=None):
    self.X = np.copy(X)  # K x d matrix of arm features
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
        
    self.theta = np.copy(theta)  # model parameter
    self.sigma = sigma  # reward noise
    self.labels = labels
    self.features = features

    self.mu = self.X.dot(self.theta)  # mean rewards of all arms
    self.randomize()

    self.best_arm = np.argmax(self.mu)  # optimal arm

  def randomize(self):
    # generate random rewards
    if self.features is None:
      self.rt = self.mu + self.sigma * np.random.randn(self.K)
    else:
      arms = np.random.choice(len(self.features), self.K, replace=False)
      if self.labels == None:
        self.X = self.features[arms]  # K x d matrix of arm features
        self.mu = self.X.dot(self.theta)  # mean rewards of all arms
        self.best_arm = np.argmax(self.mu)  # optimal arm
        self.rt = self.mu + self.sigma * np.random.randn(self.K)
        # self.rt = (np.random.rand(self.K) < self.mu).astype(float)
      else:
        self.rt = self.labels[arms]
        self.mu = self.labels[arms]
        self.X = self.features[arms]
        self.best_arm = np.argmax(self.mu)  # optimal arm

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Linear bandit: %d dimensions, %d arms" % (self.d, self.K)





class NonLinearBandit(object):
  """Non-linear bandit."""

  def __init__(self, X, theta, sigma=0.5, type='cosine', labels=None, features=None):
    self.X = np.copy(X)  # K x d matrix of arm features
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
        
    # theta can be (d,) or (m,d) for two_layer_relu8
    self.theta = np.copy(theta)
    self.sigma = sigma
    self.labels = labels
    self.features = features
    self.type = type
    if self.type == 'two_layer_relu8':
      # Expect theta shape (d,) or (12,d): [theta0; 8 hidden; 3 output] or fallback to vector
      if self.theta.ndim == 2 and self.theta.shape[0] >= 12:
        theta_hidden = self.theta[0:8]          # 8 x d
        theta_out_raw = self.theta[8:12]        # 4 x d -> 8 elements
        w_out = theta_out_raw.flatten()[:8]
      elif self.theta.ndim == 2 and self.theta.shape[0] >= 8:
        theta_hidden = self.theta[:8]
        w_out = np.ones(8)
      else:
        theta_hidden = np.tile(self.theta, (8,1))
        w_out = np.ones(8)
      Z = self.X.dot(theta_hidden.T)            # (K,8)
      self.mu = np.maximum(Z, 0).dot(w_out)     # (K,)
    elif self.type == 'quadratic':
      self.mu = (self.X.dot(self.theta))**2  # mean rewards of all arms
    elif self.type == 'cosine':
      self.sigma = 0.35
      self.mu = np.cos(3*self.X.dot(self.theta))  # mean rewards of all arms
    elif self.type == 'two_layer':
      self.mu = 2 * (self.X.dot(self.theta) * sigmoid(self.X.dot(self.theta))) + 1
    elif self.type == 'two_layer_relu':
      self.mu = 2 * np.maximum(self.X.dot(self.theta), 0) + 1
    elif self.type == 'two_layer_relu8':
      if self.theta.size == 8*self.d:
        theta_mat = self.theta.reshape(8, -1)
        w_out = np.ones(8)  # default weight
      else:
        theta_mat = np.tile(self.theta, (8,1))
        w_out = np.ones(8)
      Z = self.X.dot(theta_mat.T)
      self.mu = np.maximum(Z,0).dot(w_out)
    elif self.type == 'mlp':
      self.sigma = 0.1
      h1 = np.maximum(0, self.theta[0] * self.X[:, 0] + self.X[:, 1])
      h2 = np.maximum(0, self.X[:, 0] + self.theta[1] * self.X[:, 1])
      self.mu = h1 + h2
    elif self.type == 'fm':
      self.sigma = 0.1
      linear_term = self.X.dot(self.theta)
      interaction_coeff = self.theta[0] * self.theta[1]
      interaction_term = interaction_coeff * (self.X[:, 0] * self.X[:, 1])
      self.mu = linear_term + interaction_term
    elif self.type == 'gam':
      self.sigma = 0.1
      f1 = self.theta[0] * np.sin(np.pi * self.X[:, 0])
      f2 = self.theta[1] * np.cos(np.pi * self.X[:, 1])
      self.mu = f1 + f2
    elif self.type == 'ga2m':
      self.sigma = 0.1
      main_effect = self.theta[0] * (np.sin(np.pi * self.X[:, 0]) + np.cos(np.pi * self.X[:, 1]))
      interaction = np.sin(self.theta[1] * self.X[:, 0] * self.X[:, 1] * 5.0)
      self.mu = main_effect + interaction
    else:
      raise Exception("Unknown type.")
    self.randomize()

    self.best_arm = np.argmax(self.mu)  # optimal arm
  
  def randomize(self):
    # generate random rewards
    if self.features is None:
      self.rt = self.mu + self.sigma * np.random.randn(self.K)
    else:
      arms = np.random.choice(len(self.features), self.K, replace=False)
      if self.labels == None:
        self.X = self.features[arms]  # K x d matrix of arm features
        self.mu = self.X.dot(self.theta)  # mean rewards of all arms
        self.best_arm = np.argmax(self.mu)  # optimal arm
        self.rt = self.mu + self.sigma * np.random.randn(self.K)
        # self.rt = (np.random.rand(self.K) < self.mu).astype(float)
      else:
        self.rt = self.labels[arms]
        self.mu = self.labels[arms]
        self.X = self.features[arms]
        self.best_arm = np.argmax(self.mu)  # optimal arm

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Non-linear bandit: %d dimensions, %d arms, %s type" % (self.d, self.K, self.type)
      





def evaluate_one(Alg, params, env, n, period_size=1, return_logs=True):
  """One run of a bandit algorithm."""
  alg = Alg(env, n, params)

  regret = np.zeros(n // period_size)
  for t in range(n):
    # generate state
    env.randomize()

    # take action and update agent
    arm = alg.get_arm(t)
    alg.update(t, arm, env.reward(arm))

    # track performance
    regret_at_t = env.regret(arm)
    regret[t // period_size] += regret_at_t

  if return_logs:
    return regret, alg
  else:
    return regret


def evaluate(Alg, params, env, n=1000, period_size=1, printout=True, return_logs=True, n_jobs=-1):
  """Multiple runs of a bandit algorithm."""
  if printout:
    print("Evaluating %s" % Alg.print(), end="")
  start = time.time()

  num_exps = len(env)
  regret = np.zeros((n // period_size, num_exps))
  alg = num_exps * [None]


  if n_jobs == 1:
    output = [evaluate_one(Alg, params, env[ex], n, period_size, return_logs)
      for ex in range(num_exps)]
  else:
    output = Parallel(n_jobs=n_jobs)(delayed(evaluate_one)(Alg, params, env[ex], n, period_size, return_logs)
      for ex in range(num_exps))

  if return_logs:
      for ex in range(num_exps):
        regret[:, ex] = output[ex][0]
        alg[ex] = output[ex][1]
  else:
      for ex in range(num_exps):
        regret[:, ex] = output[ex]

  if printout:
    print(" %.1f seconds" % (time.time() - start))

  if printout:
    total_regret = regret.sum(axis=0)
    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
      np.median(total_regret), total_regret.max(), total_regret.min()))

  if return_logs:
      return regret, alg
  else:
      return regret


# Bandit algorithms

class LinBanditAlg:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.theta0 = np.zeros(self.d)  # prior mean of the model parameter
    self.Sigma0 = np.eye(self.d)  # prior covariance of the model parameter
    self.sigma = 0.5  # reward noise

    # override default values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # sufficient statistics
    self.Lambda = np.linalg.inv(self.Sigma0)
    self.B = self.Lambda.dot(self.theta0)

  def update(self, t, arm, r):
    # update sufficient statistics
    x = self.env.X[arm, :]
    self.Lambda += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)


class LinTS(LinBanditAlg):
  def get_arm(self, t):
    # linear model posterior
    Sigma_hat = np.linalg.inv(self.Lambda)
    theta_hat = Sigma_hat.dot(self.B)

    # posterior sampling
    self.theta_tilde = np.random.multivariate_normal(theta_hat, posify_mtx(Sigma_hat))
    self.mu = self.env.X.dot(self.theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinTS"


class LinUCB(LinBanditAlg):
  def __init__(self, env, n, params):
    LinBanditAlg.__init__(self, env, n, params)

    self.cew = self.confidence_ellipsoid_width(n)

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    delta = 1 / self.n
    L = np.amax(np.linalg.norm(self.env.X, axis=1))
    Lambda = np.square(self.sigma) * np.linalg.eigvalsh(np.linalg.inv(self.Sigma0)).max()  # V = \sigma^2 (posterior covariance)^{-1}
    R = self.sigma
    S = np.sqrt(self.d)
    width = np.sqrt(Lambda) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
    return width

  def get_arm(self, t):
    # linear model posterior
    Sigmahat = np.linalg.inv(self.Lambda)
    thetahat = Sigmahat.dot(self.B)

    # UCBs
    invV = Sigmahat / np.square(self.sigma)  # V^{-1} = posterior covariance / \sigma^2
    self.mu = self.env.X.dot(thetahat) + self.cew * \
      np.sqrt((self.env.X.dot(invV) * self.env.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinUCB"


class LinGreedy(LinBanditAlg):
  def get_arm(self, t):
    self.mu = np.zeros(self.K)

    # roughly 5% exploration rate
    if np.random.rand() < 0.05 * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.inf
    else:
      theta = np.linalg.solve(self.Lambda, self.B)
      self.mu = self.env.X.dot(theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Linear e-greedy"


class MixTS:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.num_mix = 2  # number of mixture components
    self.p0 = np.ones(self.num_mix) / self.num_mix  # mixture weights
    self.theta0 = np.zeros((self.num_mix, self.d))  # prior means of the mixture components
    self.Sigma0 = np.zeros((self.num_mix, self.d, self.d))  # prior covariances of the mixture components
    for i in range(self.num_mix):
      self.Sigma0[i, :, :] = np.eye(self.d)
    self.sigma = 0.5  # reward noise

    # override default values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if self.p0.ndim == 2:
      ndx = np.random.randint(self.p0.shape[0])
      self.p0 = self.p0[ndx, :]
      self.theta0 = self.theta0[ndx, :, :]
      self.Sigma0 = self.Sigma0[ndx, :, :, :]

    self.num_mix = self.p0.size

    # initialize mixture-component algorithms
    self.algs = []
    for i in range(self.num_mix):
      alg = LinTS(self.env, self.n,
        {"theta0": self.theta0[i, :], "Sigma0": self.Sigma0[i, :, :], "sigma": self.sigma})
      self.algs.append(alg)

  def update(self, t, arm, r):
    # update mixture-component algorithms
    for i in range(self.num_mix):
      self.algs[i].update(t, arm, r)

  def get_arm(self, t):
    # latent state posterior
    _, prior_logdet = np.linalg.slogdet(self.Sigma0)
    Lambda0 = np.linalg.inv(self.Sigma0)

    logp = np.zeros(self.num_mix)
    for i in range(self.num_mix):
      post_cov = np.linalg.inv(self.algs[i].Lambda)
      _, post_logdet = np.linalg.slogdet(post_cov)
      logp[i] = 0.5 * ((self.d * np.log(2 * np.pi) + post_logdet) -
        ((self.d + t) * np.log(2 * np.pi) + 2 * t * np.log(self.sigma) + prior_logdet[i])) + \
        0.5 * (self.algs[i].B.T.dot(post_cov).dot(self.algs[i].B) -
        self.theta0[i, :].T.dot(Lambda0[i, :, :]).dot(self.theta0[i, :])) + \
        np.log(self.p0[i])

    self.p = np.exp(logp - logp.max())
    self.p /= self.p.sum()

    # posterior sampling
    self.component_tilde = np.random.choice(self.num_mix, p=self.p)
    arm = self.algs[self.component_tilde].get_arm(t)

    return arm

  @staticmethod
  def print():
    return "MixTS"


class LinDiffTS(LinBanditAlg):
  def __init__(self, env, n, params):
    LinBanditAlg.__init__(self, env, n, params)

  def map_estimator(self, theta0, Sigma0, t):
    Lambda0 = np.linalg.inv(Sigma0)
    Lambda = Lambda0 + self.Lambda
    Sigma = np.linalg.inv(Lambda)
    theta = Sigma.dot(Lambda0.dot(theta0) + self.B)
    return theta, Sigma

  def get_arm(self, t):
    # posterior sampling through likelihood
    map_lambda = lambda theta0, Sigma0: self.map_estimator(theta0, Sigma0, t)
    self.theta_tilde = self.prior.posterior_sample_map(map_lambda)[0, :]
    self.mu = self.env.X.dot(self.theta_tilde)

    # # evidence
    # Sigma_bar = np.linalg.inv(self.Lambda)
    # theta_bar = Sigma_bar.dot(self.B)

    # # posterior sampling
    # theta_tilde = self.prior.posterior_sample(theta_bar, posify_mtx(Sigma_bar))[0, :]
    # self.mu = self.env.X.dot(theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffTS"


class LinDiffTSChung:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.sigma = 0.5  # reward noise

    # override default values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # sufficient statistics
    self.Xh = np.zeros((self.n, self.d))  # history of feature vectors
    self.yh = np.zeros(self.n)  # history of rewards

  def update(self, t, arm, r):
    # update sufficient statistics
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def loglik_grad(self, theta0, t, linear_growth=False):
    if t == 0:
      grad = np.zeros(self.d)
    else:
      X = self.Xh[: t, :]
      y = self.yh[: t]
      v = X.T.dot(y - X.dot(theta0)) / np.linalg.norm(y - X.dot(theta0))
      grad = v / np.square(self.sigma)
      if linear_growth:
        grad *= np.sqrt(t)
    return grad

  def get_arm(self, t):
    # posterior sampling through loglik gradient
    grad_lambda = lambda theta0: self.loglik_grad(theta0, t)
    self.theta_tilde = self.prior.posterior_sample_grad(grad_lambda)[0, :]
    self.mu = self.env.X.dot(self.theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffTSChung"



class LinDiffDPTS:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.Xh = np.zeros((self.n, self.d))  # history of feature vectors
    self.yh = np.zeros(self.n)  # history of rewards
    self.sigma = 0.5  # reward noise
    # SGLD hyperparameters
    self.num_steps_sgld = 1
    self.step_size_sgld = 0.05
    self.noise_scale = 0.01

    # override default values (e.g., prior, sigma)
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # Ensure attributes exist even if not passed
    if not hasattr(self, 'num_steps_sgld'):
      self.num_steps_sgld = 1
    if not hasattr(self, 'step_size_sgld'):
      self.step_size_sgld = 0.05
    if not hasattr(self, 'noise_scale'):
      self.noise_scale = 0.01

  def update(self, t, arm, r):
    # update sufficient statistics
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    # posterior sampling using diffusion SGLD prior
    self.theta_tilde = self.prior.posterior_sample_dpts(
        self.Xh, self.yh,
        num_steps_sgld=self.num_steps_sgld,
        step_size_sgld=self.step_size_sgld,
        noise_scale=self.noise_scale)[0, :]
    self.mu = self.env.X.dot(self.theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffDPTS"



class LinDiffDPS:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # feature dimension
    self.n = n  # horizon

    # History buffers (filled incrementally)
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0  # reward noise std
    self.eta = 0.05   # likelihood-drift step size

    # Override defaults with user-provided params (e.g., prior, sigma, eta)
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # Sanity: ensure required attributes exist
    if not hasattr(self, 'prior'):
      raise ValueError("LinDPS requires a 'prior' instance (DiffusionPrior)")

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    self.theta_tilde = self.prior.posterior_sample_dps(
        self.Xh, self.yh,
        sigma=self.sigma,
        eta=self.eta)[0, :]

    # Compute expected rewards μ_k = x_k · θ and pick the best arm
    self.mu = self.env.X.dot(self.theta_tilde)
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffDPS"


# ---------------------------------------------------------------------------
#                      Diffusion MAP Estimation (DMAP)
# ---------------------------------------------------------------------------


class LinDiffDMAP:
  """Linear bandit algorithm using Diffusion MAP posterior mode (Algorithm-3)."""

  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.K
    self.d = self.env.d
    self.n = n

    # History buffers
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0   # observation noise std
    self.K_inner = 10  # GD steps per diffusion layer
    self.eta = 0.01    # learning rate

    # Override with user-provided values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if not hasattr(self, 'prior'):
      raise ValueError("LinDiffDMAP requires a 'prior' (DiffusionPrior) instance")

  # ---------------------- interaction loop ------------------------------

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    theta_map = self.prior.posterior_sample_dmap(
        self.Xh, self.yh,
        sigma=self.sigma,
        K_inner=self.K_inner,
        eta=self.eta)[0, :]

    self.theta_tilde = theta_map  # keep same attribute name as others for consistency
    self.mu = self.env.X.dot(self.theta_tilde)
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffDMAP"



class LinDiffDMAPNew:
  """Linear bandit algorithm using Diffusion MAP posterior mode (Algorithm-3)."""

  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.K
    self.d = self.env.d
    self.n = n

    # History buffers
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0   # observation noise std
    self.K_inner = 10  # GD steps per diffusion layer
    self.eta = 0.01    # learning rate

    # Override with user-provided values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if not hasattr(self, 'prior'):
      raise ValueError("LinDiffDMAPNew requires a 'prior' (DiffusionPrior) instance")

  # ---------------------- interaction loop ------------------------------

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    theta_map = self.prior.posterior_sample_dmap_new(
        self.Xh, self.yh,
        sigma=self.sigma,
        K_inner=self.K_inner,
        eta=self.eta)[0, :]

    self.theta_tilde = theta_map  # keep same attribute name as others for consistency
    self.mu = self.env.X.dot(self.theta_tilde)
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffDMAPNew"



class LinDiffDPSACR:

  """Linear bandit using Diffusion Posterior Sampling with Action-Conditioned Reweighting."""

  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.K
    self.d = self.env.d
    self.n = n

    # History buffers for past interactions
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Hyper-parameters
    self.sigma = 1.0  # reward noise std
    self.eta = 0.05   # likelihood drift step size
    self.h = 1.0      # RBF bandwidth for ACR weighting

    # Override with user-provided params (incl. prior)
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if not hasattr(self, 'prior'):
      raise ValueError("LinDiffDPSACR requires 'prior' (DiffusionPrior) instance")

  # ---------------- interaction ----------------

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    # Obtain current arms' contexts (assumed fixed across time)
    Xt_cur = self.env.X  # shape (K,d)

    theta_sample = self.prior.posterior_sample_dps_acr(
        self.Xh, self.yh, Xt_cur,
        sigma=self.sigma,
        eta=self.eta,
        h=self.h)[0, :]

    self.theta_tilde = theta_sample
    self.mu = Xt_cur.dot(self.theta_tilde)
    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinDiffDPSACR"


class LogBanditAlg:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.theta0 = np.zeros(self.d)  # prior mean of the model parameter
    self.Sigma0 = np.eye(self.d)  # prior covariance of the model parameter

    # override default values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # sufficient statistics
    self.Lambda0 = np.linalg.inv(self.Sigma0)
    self.Xh = np.zeros((self.n, self.d))  # history of feature vectors
    self.yh = np.zeros(self.n)  # history of rewards

    self.irls_theta = np.zeros(self.d)

  def update(self, t, arm, r):
    # update sufficient statistics
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def solve(self, t):
    theta, Hessian, converged = irls(
      self.Xh[: t, :], self.yh[: t], self.theta0, self.Lambda0, np.copy(self.irls_theta))
    if converged:
      self.irls_theta = np.copy(theta)
    else:
      self.irls_theta = np.zeros(self.d)

    return theta, Hessian, converged


class LogGreedy(LogBanditAlg):

  def get_arm(self, t):
    """Epsilon-greedy algorithm for logistic bandits.

    Uses the current IRLS estimate of θ (Laplace mode) to compute mean
    rewards μ_k = σ(x_k^⊤ θ̂).  With a small exploration probability that
    decays as O(1/√t) we instead pick a random arm.
    """

    # --- Exploration schedule (≈ 5 % initially, decays with 1/√t) ---
    eps = 0.05 * np.sqrt(self.n / (t + 1)) / 2.0
    if np.random.rand() < eps:
      # explore: force a random arm to look best
      self.mu = np.zeros(self.K)
      self.mu[np.random.randint(self.K)] = np.inf
    else:
      # exploit: fit logistic regression to history and pick best arm
      theta_hat, _, _ = self.solve(t)
      z = self.env.X.dot(theta_hat)  # linear scores
      self.mu = sigmoid(z)           # Bernoulli means via logistic link

    arm = int(np.argmax(self.mu))
    return arm




class LogTS(LogBanditAlg):
  def get_arm(self, t):
    # Laplace posterior approximation
    theta_hat, Hessian, _ = self.solve(t)
    Sigma_hat = np.linalg.inv(Hessian)

    # posterior sampling
    theta_tilde = np.random.multivariate_normal(theta_hat, Sigma_hat)
    self.mu = self.env.X.dot(theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LogTS"


class LogDiffTS(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)

  def map_estimator(self, theta0, Sigma0, t):
    theta, Lambda, _ = irls(
      self.Xh[: t, :], self.yh[: t], theta0, np.linalg.inv(Sigma0), np.zeros(self.d))
    Sigma = np.linalg.inv(Lambda)
    return theta, Sigma

  def get_arm(self, t):
    # posterior sampling through likelihood
    map_lambda = lambda theta0, Sigma0: self.map_estimator(theta0, Sigma0, t)
    theta_tilde = self.prior.posterior_sample_map(map_lambda)[0, :]
    self.mu = self.env.X.dot(theta_tilde)

    # # evidence
    # theta_bar, Hessian, _ = self.solve(t)
    # Sigma_bar = np.linalg.inv(Hessian)

    # # posterior sampling
    # theta_tilde = self.prior.posterior_sample(theta_bar, posify_mtx(Sigma_bar))[0, :]
    # self.mu = self.env.X.dot(theta_tilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LogDiffTS"



class LogDiffDPTS(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)
    self.sigma = params.get('sigma', 1.0)

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    # ----- Logistic link (sigmoid) -----
    def sigmoid_predict(theta_vec, X_tensor):
      return torch.sigmoid(X_tensor.matmul(theta_vec))

    theta_sample = self.prior.posterior_sample_dpts(
        self.Xh, self.yh,
        link_func=torch.sigmoid,  # for linear path in _sgld_appr
        sigma=self.sigma,
        num_steps_sgld=self.num_steps_sgld,
        step_size_sgld=self.step_size_sgld,
        noise_scale=self.noise_scale,
        predict_func=sigmoid_predict)[0, :]

    # Expected reward for each arm under logistic model
    z = self.env.X.dot(theta_sample)
    self.mu = 1.0 / (1.0 + np.exp(-z))

    arm = np.argmax(self.mu)
    self.theta_tilde = theta_sample
    return arm

  @staticmethod
  def print():
    return "LogDiffDPTS"



class LogDiffDPS(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)
    self.eta = params.get('eta', 0.05)
    self.sigma = params.get('sigma', 1.0)

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    def sigmoid_predict(theta_vec, X_tensor):
      return torch.sigmoid(X_tensor.matmul(theta_vec))

    theta_sample = self.prior.posterior_sample_dps(
        self.Xh, self.yh,
        sigma=self.sigma,
        eta=self.eta,
        predict_func=sigmoid_predict)[0, :]

    z = self.env.X.dot(theta_sample)
    self.mu = 1.0 / (1.0 + np.exp(-z))

    arm = np.argmax(self.mu)
    self.theta_tilde = theta_sample
    return arm

  @staticmethod
  def print():
    return "LogDiffDPS"



class LogDiffDMAP(LogBanditAlg):
  def __init__(self, env, n, params):
    LogBanditAlg.__init__(self, env, n, params)
    self.sigma = params.get('sigma', 1.0)
    self.K_inner = params.get('K_inner', 10)
    self.eta = params.get('eta', 0.01)
    
    if not hasattr(self, 'prior'):
      raise ValueError("LogDiffDMAP requires a 'prior' instance (DiffusionPrior)")

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r
    
  def get_arm(self, t):
    theta_map = self.prior.posterior_sample_dmap(
        self.Xh, self.yh,
        sigma=self.sigma,
        K_inner=self.K_inner,
        eta=self.eta)[0, :]
    self.theta_tilde = theta_map
    self.mu = self.env.X.dot(self.theta_tilde)
    arm = np.argmax(self.mu)
    return arm
    
    
  @staticmethod
  def print():
    return "LogDiffDMAP"


class NeuralDiffDPTS:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.Xh = np.zeros((self.n, self.d))  # history of feature vectors
    self.yh = np.zeros(self.n)  # history of rewards
    self.sigma = 0.5  # reward noise
    # SGLD hyperparameters
    self.num_steps_sgld = 1
    self.step_size_sgld = 0.05
    self.noise_scale = 0.01

    # override default values (e.g., prior, sigma)
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # Ensure attributes exist even if not passed
    if not hasattr(self, 'num_steps_sgld'):
      self.num_steps_sgld = 1
    if not hasattr(self, 'step_size_sgld'):
      self.step_size_sgld = 0.05
    if not hasattr(self, 'noise_scale'):
      self.noise_scale = 0.01

  def update(self, t, arm, r):
    # update sufficient statistics
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    # ---------- define MLP predict func ----------
    hidden = 32  # must match Synthetic.SmallMLP
    def predict_mlp(theta_vec, X_tensor):
      return self.prior._mlp_forward_torch(theta_vec, X_tensor, self.d, hidden)

    # posterior sampling with neural likelihood
    theta_sample = self.prior.posterior_sample_dpts(
        self.Xh, self.yh,
        sigma=self.sigma,
        num_steps_sgld=self.num_steps_sgld,
        step_size_sgld=self.step_size_sgld,
        noise_scale=self.noise_scale,
        predict_func=predict_mlp)[0, :]

    # reward prediction for each arm via same MLP
    Xt = torch.from_numpy(self.env.X).double()
    mu_torch = predict_mlp(torch.from_numpy(theta_sample).double(), Xt)
    self.mu = mu_torch.detach().cpu().numpy()

    arm = np.argmax(self.mu)
    self.theta_tilde = theta_sample
    return arm

  @staticmethod
  def print():
    return "LinDiffDPTS"



class NeuralDiffDPS:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # feature dimension
    self.n = n  # horizon

    # History buffers (filled incrementally)
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0  # reward noise std
    self.eta = 0.05   # likelihood-drift step size

    # Override defaults with user-provided params (e.g., prior, sigma, eta)
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # Sanity: ensure required attributes exist
    if not hasattr(self, 'prior'):
      raise ValueError("LinDPS requires a 'prior' instance (DiffusionPrior)")

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    hidden = 32
    def predict_mlp(theta_vec, X_tensor):
      return self.prior._mlp_forward_torch(theta_vec, X_tensor, self.d, hidden)

    theta_sample = self.prior.posterior_sample_dps(
        self.Xh, self.yh,
        sigma=self.sigma,
        eta=self.eta,
        predict_func=predict_mlp)[0, :]

    Xt = torch.from_numpy(self.env.X).double()
    mu_torch = predict_mlp(torch.from_numpy(theta_sample).double(), Xt)
    self.mu = mu_torch.detach().cpu().numpy()
    self.theta_tilde = theta_sample
    arm = int(np.argmax(self.mu))
    return arm

  @staticmethod
  def print():
    return "LinDiffDPS"


# ---------------------------------------------------------------------------
#                      Diffusion MAP Estimation (DMAP)
# ---------------------------------------------------------------------------


class NeuralDiffDMAP:
  """Linear bandit algorithm using Diffusion MAP posterior mode (Algorithm-3)."""

  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.K
    self.d = self.env.d
    self.n = n

    # History buffers
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0   # observation noise std
    self.K_inner = 10  # GD steps per diffusion layer
    self.eta = 0.01    # learning rate

    # Override with user-provided values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if not hasattr(self, 'prior'):
      raise ValueError("LinDiffDMAP requires a 'prior' (DiffusionPrior) instance")

  # ---------------------- interaction loop ------------------------------

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    hidden = 32
    def predict_mlp(theta_vec, X_tensor):
      return self.prior._mlp_forward_torch(theta_vec, X_tensor, self.d, hidden)

    theta_map = self.prior.posterior_sample_dmap(
        self.Xh, self.yh,
        sigma=self.sigma,
        K_inner=self.K_inner,
        eta=self.eta,
        predict_func=predict_mlp)[0, :]

    Xt = torch.from_numpy(self.env.X).double()
    mu_torch = predict_mlp(torch.from_numpy(theta_map).double(), Xt)
    self.mu = mu_torch.detach().cpu().numpy()
    self.theta_tilde = theta_map
# return chosen arm
    arm = int(np.argmax(self.mu))
    return arm

  @staticmethod
  def print():
    return "LinDiffDMAP"





class NeuralDiffDMAPNew:
  """Linear bandit algorithm using Diffusion MAP posterior mode (Algorithm-3)."""

  def __init__(self, env, n, params):
    self.env = env
    self.K = self.env.K
    self.d = self.env.d
    self.n = n

    # History buffers
    self.Xh = np.zeros((self.n, self.d))
    self.yh = np.zeros(self.n)

    # Default hyper-parameters
    self.sigma = 1.0   # observation noise std
    self.K_inner = 10  # GD steps per diffusion layer
    self.eta = 0.01    # learning rate

    # Override with user-provided values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    if not hasattr(self, 'prior'):
      raise ValueError("LinDiffDMAP requires a 'prior' (DiffusionPrior) instance")

  # ---------------------- interaction loop ------------------------------

  def update(self, t, arm, r):
    self.Xh[t, :] = self.env.X[arm, :]
    self.yh[t] = r

  def get_arm(self, t):
    hidden = 32
    def predict_mlp(theta_vec, X_tensor):
      return self.prior._mlp_forward_torch(theta_vec, X_tensor, self.d, hidden)

    theta_map = self.prior.posterior_sample_dmap_new(
        self.Xh, self.yh,
        sigma=self.sigma,
        K_inner=self.K_inner,
        eta=self.eta,
        predict_func=predict_mlp)[0, :]

    Xt = torch.from_numpy(self.env.X).double()
    mu_torch = predict_mlp(torch.from_numpy(theta_map).double(), Xt)
    self.mu = mu_torch.detach().cpu().numpy()
    self.theta_tilde = theta_map
# return chosen arm
    arm = int(np.argmax(self.mu))
    return arm

  @staticmethod
  def print():
    return "LinDiffDMAPNew"





class NeuralTS:

    """Neural Thompson Sampling for contextual bandits.

    Parameters
    ----------
    num_arm : int
        Number of arms.
    dim_context : int
        Dimension of context feature vector for each arm.
    model : torch.nn.Module
        Prediction model that maps a context to a scalar reward estimate.
    optimizer : torch.optim.Optimizer
        Optimizer used to train the model.
    criterion : callable
        Loss function (e.g., torch.nn.MSELoss).
    collector : object
        Helper that must expose ``collect_data(context, arm, reward)`` and ``fetch_batch()``.
    nu : float
        Exploration variance parameter (see original NeuralTS paper).
    batch_size : int, optional
        Minibatch size used when ``step >= batch_size``; if None uses full batch every update.
    image : bool, default False
        If True, context is treated as image (expects BCHW tensor per arm).
    reduce : int, optional
        Update the network every ``reduce`` calls to ``update_model``.
    reg : float, default 1.0
        Ridge regularisation weight for the design matrix.
    device : str, default 'cpu'
        Device on which to run computations.
    name : str, default 'NeuralTS'
        Identifier.
    """

    def __init__(self, env, n, params):
        """Neural Thompson Sampling for contextual bandits (compatible with evaluate_one).

        Parameters
        ----------
        env : object
            Environment that exposes X (arm contexts), K (number of arms) and d (context dimension).
        n : int
            Horizon length (kept for API compatibility but not directly used).
        params : dict
            Dictionary containing hyper-parameters required by the original NeuralTS implementation. Must include
            keys ``model``, ``optimizer``, ``criterion``, ``collector`` and ``nu``. Optional keys are
            ``batch_size``, ``image``, ``reduce``, ``reg``, ``device`` and ``name``.
        """
        # Save environment and horizon (the latter stored only for completeness)
        self.env = env
        self.n = n

        # When explicit neural components are not provided, build defaults
        if "model" in params:
            model = params["model"]
            optimizer = params["optimizer"]
            criterion = params["criterion"]
            collector = params["collector"]
            nu = params.get("nu", 0.01)
        else:
            hidden = params.get("hidden", 32)
            dim_in = env.d
            class _SmallMLP(nn.Module):
                def __init__(self, d_in):
                    super().__init__()
                    self.fc1 = nn.Linear(d_in, hidden)
                    self.act = nn.Tanh()
                    self.fc2 = nn.Linear(hidden, 1)
                def forward(self, x):
                    return self.fc2(self.act(self.fc1(x.float())))
            model = _SmallMLP(dim_in)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            criterion = nn.MSELoss()
            collector = Collector()
            nu = params.get("nu", 0.01)

        # ---- optional parameters with defaults ----
        batch_size = params.get("batch_size")
        image = params.get("image", False)
        reduce = params.get("reduce")
        reg = params.get("reg", 1.0)
        device = params.get("device", "cpu")
        name = params.get("name", "NeuralTS")

        # ------------------------------------------------------------------
        # Original NeuralTS initialisation logic
        # ------------------------------------------------------------------
        self.image = image
        self.reduce = reduce
        self.num_arm = env.K
        self.dim_context = env.d

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion

        if batch_size is not None:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None

        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    # ------------------------------------------------------------------
    def clear(self):
        """Reset history and design vector."""
        self.collector.clear()
        self.Design = self.reg * torch.ones(self.num_params, device=self.device)
        self.last_cxt = None
        self.step = 0

    # ------------------------------------------------------------------
    def get_arm(self, t):
        """Select an arm according to Thompson sampling.

        Parameters
        ----------
        context : torch.Tensor
            Tensor containing the context for every arm. Shape depends on
            ``image`` flag:
                * If False: (K, d)
                * If True:  (K, C, H, W)
        """
        rewards = []
        context = torch.from_numpy(self.env.X).to(self.device).double()

        for i in range(self.num_arm):
            self.model.zero_grad()

            # Forward pass for a single arm
            if self.image:
                ri = self.model(context[i:i + 1, :, :, :])
            else:
                ri = self.model(context[i])

            # Compute gradient wrt model params
            ri.backward()
            grad = torch.cat([p.grad.contiguous().view(-1).detach() for p in self.model.parameters()])

            # Compute posterior std following NeuralTS derivation
            squared_sigma = self.reg * self.nu * grad * grad / self.Design
            sigma = torch.sqrt(torch.sum(squared_sigma))

            # Thompson sample
            sample_r = ri + torch.randn(1, device=self.device) * sigma
            rewards.append(sample_r.item())

        arm_to_pull = int(np.argmax(rewards))
        return arm_to_pull

    # ------------------------------------------------------------------
    def update(self, t, arm, r):
        context = torch.from_numpy(self.env.X[arm, :]).to(self.device).double()
        reward = r
        """Record observed transition."""
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context
        self.update_model(num_iter=10)


    # ------------------------------------------------------------------
    def update_model(self, num_iter: int):
        """Update neural network parameters.

        Parameters
        ----------
        num_iter : int
            Number of optimisation steps to perform.
        """
        self.step += 1

        # Optionally subsample update frequency
        if self.reduce and self.step % self.reduce != 0:
            return

        # Weight decay schedule (follows original paper)
        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # -------------- gradient update --------------
        if self.batchsize is not None and self.batchsize < self.step:
            # Mini-batch SGD using sampler
            ploader = sample_data(self.loader)
            for _ in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is NaN!'
        else:
            # Full-batch update on all collected data
            contexts, rewards = self.collector.fetch_batch()
            if not isinstance(contexts, torch.Tensor):
                contexts = torch.stack(contexts, dim=0)
            contexts = contexts.to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            self.model.train()
            for _ in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is NaN!'

        # -------------- update design vector --------------
        self.model.zero_grad()
        if self.image:
            re = self.model(self.last_cxt.unsqueeze(0))
        else:
            re = self.model(self.last_cxt)
        re.backward()
        grad = torch.cat([p.grad.contiguous().view(-1).detach() for p in self.model.parameters()])
        self.Design += grad * grad
    
    @staticmethod
    def print():
        return "NeuralTS"


class NeuralUCB:
    """NeuralUCB algorithm compatible with evaluate_one framework.

    This implementation mirrors NeuralTS but uses the UCB-style exploration term
    derived in the original NeuralUCB paper (squared confidence bound instead of
    random Thompson sampling).
    """
    # ------------------------------------------------------------------
    def __init__(self, env, n, params):
        """Parameters follow the same convention as NeuralTS.__init__."""
        # Save environment / horizon
        self.env = env
        self.n = n

        # When explicit neural components are not provided, build defaults
        if "model" in params:
            model = params["model"]
            optimizer = params["optimizer"]
            criterion = params["criterion"]
            collector = params["collector"]
            nu = params.get("nu",0.01)
        else:
            hidden = params.get("hidden", 32)
            dim_in = env.d
            class _SmallMLP(nn.Module):
                def __init__(self, d_in):
                    super().__init__()
                    self.fc1 = nn.Linear(d_in, hidden)
                    self.act = nn.Tanh()
                    self.fc2 = nn.Linear(hidden, 1)
                def forward(self, x):
                    return self.fc2(self.act(self.fc1(x.float())))
            model = _SmallMLP(dim_in)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            criterion = nn.MSELoss()
            collector = Collector()
            nu = params.get("nu",0.01)

        # ---- optional parameters with defaults ----
        batch_size = params.get("batch_size")
        image = params.get("image", False)
        reduce = params.get("reduce")
        reg = params.get("reg", 1.0)
        device = params.get("device", "cpu")
        name = params.get("name", "NeuralUCB")

        # Store basics
        self.image = image
        self.reduce = reduce
        self.num_arm = env.K
        self.dim_context = env.d
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        if batch_size is not None:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0
        self.num_params = get_param_size(model)
        self.clear()

    # ------------------------------------------------------------------
    def clear(self):
        """Reset history and design vector."""
        self.collector.clear()
        self.Design = self.reg * torch.ones(self.num_params, device=self.device)
        self.last_cxt = None
        self.step = 0

    # ------------------------------------------------------------------
    def get_arm(self, t):
        """Select arm using NeuralUCB criterion."""
        rewards = []
        grad_list = []
        context = torch.from_numpy(self.env.X).to(self.device).double()
        for i in range(self.num_arm):
            self.model.zero_grad()
            if self.image:
                ri = self.model(context[i:i + 1, :, :, :])
            else:
                ri = self.model(context[i])
            ri.backward()
            grad = torch.cat([p.grad.contiguous().view(-1).detach() for p in self.model.parameters()])
            grad_list.append(grad)
            squared_sigma = self.reg * self.nu * grad * grad / self.Design
            sigma = torch.sqrt(torch.sum(squared_sigma))
            # UCB estimate (mean + confidence width)
            sample_r = ri + sigma
            rewards.append(sample_r.item())
        arm_to_pull = int(np.argmax(rewards))
        # Update design matrix after choosing the arm (as in original code)
        self.Design += grad_list[arm_to_pull] * grad_list[arm_to_pull]
        return arm_to_pull

    # ------------------------------------------------------------------
    def update(self, t, arm, r):
        """Record transition and train network."""
        context = torch.from_numpy(self.env.X[arm, :]).to(self.device).double()
        reward = r
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context
        self.update_model(num_iter=10)

    # ------------------------------------------------------------------
    def update_model(self, num_iter: int):
        """Identical to NeuralTS.update_model but without early exit changes."""
        self.step += 1
        if self.reduce and self.step % self.reduce != 0:
            return
        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step
        if self.batchsize is not None and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for _ in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is NaN!'
        else:
            contexts, rewards = self.collector.fetch_batch()
            if not isinstance(contexts, torch.Tensor):
                contexts = torch.stack(contexts, dim=0)
            contexts = contexts.to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for _ in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is NaN!'

    # ------------------------------------------------------------------
    @staticmethod
    def print():
        return "NeuralUCB"


# ------------------ helper functions ------------------


def get_param_size(model: torch.nn.Module) -> int:
    """Return total number of parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def sample_data(loader):
    """Infinite generator over a torch DataLoader."""
    while True:
        for batch in loader:
            yield batch

class Collector(Dataset):
    '''
    Collect the context vectors that have appeared 
    '''

    def __init__(self):
        super(Collector, self).__init__()
        self.context = []
        self.rewards = []
        self.chosen_arms = []

    def __getitem__(self, key):
        return self.context[key], self.rewards[key]

    def __len__(self):
        return len(self.rewards)

    def collect_data(self, context, arm, reward):
        self.context.append(context.cpu())
        self.chosen_arms.append(arm)
        self.rewards.append(reward)

    def fetch_batch(self, batch_size=None):
        if batch_size is None or batch_size > len(self.rewards):
            return self.context, self.rewards
        else:
            offset = np.random.randint(0, len(self.rewards) - batch_size)
            return self.context[offset:offset + batch_size], self.rewards[offset: offset + batch_size]

    def clear(self):
        self.context = []
        self.rewards = []
        self.chosen_arms = []