# Imports and defaults
from bandit import *
from diffusion_posterior import *
import joblib
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import time
import argparse
import os
import random

mpl.style.use("classic")
mpl.rcParams["figure.figsize"] = [6, 4]

mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["errorbar.capsize"] = 3
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["legend.fontsize"] = 16

mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0

import platform
print("python %s" % platform.python_version())
print("matplotlib %s" % mpl.__version__)
print("%d joblib CPUs" % joblib.cpu_count())

import warnings
warnings.filterwarnings("ignore")

def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (0.5, 2.5)
  else:
    return (None, None)




parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="cross")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)


d = 2  # number of features
K = 100  # number of arms
n = 2000  # horizon
num_runs = 64  # number of random runs per super run
num_super_runs = 1  # priors are re-estimated
reward_model = "linear"  # reward model
sigma = 1.0  # reward noise
n_pretrain = 10000  # number of samples for estimating the prior


# Default hyperparameters (used for logistic runs or as fallbacks)
num_steps_sgld = 1
step_size_sgld = 0.05
noise_scale = 0.05
eta = 0.005
K_inner = 1
dmap_eta = 0.05


# Best-performing hyperparameters per problem (from slurm-39047161 logs)
LIN_DIFF_DPTS_PARAMS = {
  "cross": {"num_steps_sgld": 1, "step_size_sgld": 0.1, "noise_scale": 0.1},
  "rays": {"num_steps_sgld": 1, "step_size_sgld": 0.1, "noise_scale": 0.005},
  "triangles": {"num_steps_sgld": 1, "step_size_sgld": 0.05, "noise_scale": 0.01},
  "swirl": {"num_steps_sgld": 10, "step_size_sgld": 0.01, "noise_scale": 0.005},
  "H": {"num_steps_sgld": 10, "step_size_sgld": 0.1, "noise_scale": 0.05},
  "corners": {"num_steps_sgld": 1, "step_size_sgld": 0.005, "noise_scale": 0.1},
}

LIN_DIFF_DPS_PARAMS = {
  "cross": 0.005,
  "rays": 0.1,
  "triangles": 0.1,
  "swirl": 0.01,
  "H": 0.005,
  "corners": 0.05,
}

LIN_DIFF_DMAP_PARAMS = {
  "cross": {"K_inner": 1, "dmap_eta": 0.1},
  "rays": {"K_inner": 1, "dmap_eta": 0.1},
  "triangles": {"K_inner": 10, "dmap_eta": 0.005},
  "swirl": {"K_inner": 1, "dmap_eta": 0.05},
  "H": {"K_inner": 10, "dmap_eta": 0.01},
  "corners": {"K_inner": 1, "dmap_eta": 0.01},
}


# determine which problem(s) to run: if '--problem all' run all predefined problems
if args.problem.lower() == "all":
  problems = ["cross", "rays", "triangles", "swirl", "H", "corners"]
else:
  problems = [args.problem]

for problem in problems:

  for super_run in range(num_super_runs):
    if problem == "cross":
      # mixture prior
      num_mix = 2  # number of mixture components
      p0 = np.ones(num_mix) / num_mix
      theta0 = np.zeros((num_mix, d))
      Sigma0 = np.zeros((num_mix, d, d))
      Sigma0[0, :, :] = np.asarray([[1, 0.99], [0.99, 1]])
      Sigma0[1, :, :] = np.asarray([[1, -0.9], [-0.9, 1]])

      # sample from the prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        component = np.random.choice(num_mix, p=p0)
        S0[i, :] = np.random.multivariate_normal(theta0[component, :], Sigma0[component, :, :])
    elif problem == "rays":
      num_mix = 2

      # sample from prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        accept = False
        while not accept:
          s0 = 6 * (np.random.rand(d) - 0.5)
          s0_norm = s0 / np.linalg.norm(s0)
          accept = (np.linalg.norm(s0) < 3) and \
            ((s0_norm.dot(np.asarray([1, 0])) > 0.95) or
            (s0_norm.dot(np.asarray([-1, 0])) > 0.95) or
            (s0_norm.dot(np.asarray([0, 1])) > 0.95) or
            (s0_norm.dot(np.asarray([0, -1])) > 0.95))
        S0[i, :] = s0
    elif problem == "triangles":
      num_mix = 2

      # sample from prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        accept = False
        while not accept:
          s0 = 6 * (np.random.rand(d) - 0.5)
          accept = ((s0[0] > 0) and (np.abs(s0[1]) < np.abs(s0[0]))) or \
            ((s0[0] < 0) and (np.abs(s0[1]) < np.abs(s0[0]) / 3))
        S0[i, :] = s0
    elif problem == "swirl":
      num_mix = 2

      # sample from prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        u = np.random.rand()
        angle = 2 * np.pi * u
        radius = 3.5 * u
        width = u
        s0 = np.asarray([np.sin(angle), np.cos(angle)]) * (radius - width * (np.random.rand() - 0.5))
        s0 += np.asarray([1, -1])
        S0[i, :] = s0
    elif problem == "H":
      num_mix = 2

      # sample from prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        accept = False
        while not accept:
          s0 = 6 * (np.random.rand(d) - 0.5)
          accept = (s0[0] < -2.5) or (s0[0] > 2.5) or ((s0[1] > -0.25) and (s0[1] < 0.25))
        S0[i, :] = s0
    elif problem == "corners":
      num_mix = 2

      # sample from prior
      S0 = np.zeros((n_pretrain, d))
      for i in range(n_pretrain):
        accept = False
        while not accept:
          s0 = 6 * (np.random.rand(d) - 0.5)
          accept = ((np.abs(s0[0]) > 2.5) and (np.abs(s0[1]) > 1)) or \
            ((np.abs(s0[1]) > 2.5) and (np.abs(s0[0]) > 1))
        S0[i, :] = s0
    else:
      raise Exception("Unknown problem.")

    S0 = S0[np.random.permutation(n_pretrain), :]
    S0_train = S0[: - num_runs, :]
    S0_test = S0[- num_runs :, :]

    # Gaussian prior approximation
    theta0_bar = S0_train.mean(axis=0)
    Delta = S0_train - theta0_bar[np.newaxis, :]
    Sigma0_bar = Delta.T.dot(Delta) / S0_train.shape[0]

    # Gaussian mixture prior approximation
    gm = GaussianMixture(n_components=num_mix, covariance_type="full").fit(S0_train)
    p0_gm = gm.weights_
    theta0_gm = gm.means_
    Sigma0_gm = gm.covariances_

    # diffusion prior learning
    T = 100
    alpha = 0.97
    prior = DiffusionPrior(d, T, alpha, reg=1e-4, hidden_size=100)
    prior.train(S0_train)
    joblib.dump(prior, "Results_linear_2000/%s_diffusion_model.pkl" % problem)

    if super_run == 0:
      subS0 = np.minimum(np.maximum(S0[: 1000, :], -10), 10)
      S = prior.sample(1000)
      S = np.minimum(np.maximum(S, -10), 10)


      plt.scatter(subS0[:, 0], subS0[:, 1], s=6, marker=".", color="b", label="True Prior")
      plt.scatter(S[0, :, 0], S[0, :, 1], s=6, marker=".", color="grey", label="Diffusion Prior")
      plt.title("Problem %s" % problem)
      plt.ylabel("True and diffusion priors")
      plt.tight_layout()
      plt.legend()
      plt.savefig("Results_linear_2000/%s_diffusion_prior.pdf" % problem)
      plt.close()

      np.save("Results_linear_2000/%s_prior.npy" % problem, subS0)
      np.save("Results_linear_2000/%s_diffusion_prior.npy" % problem, S)



      plt.scatter(subS0[:, 0], subS0[:, 1], s=6, marker=".", color="b", label="True Prior")
      plt.title("Problem %s" % problem)
      plt.ylabel("True priors")
      plt.tight_layout()
      plt.legend()
      plt.savefig("Results_linear_2000/%s_prior.pdf" % problem)
      plt.close()

      # debug placeholder removed

    # bandit environments
    envs = []
    for run in range(num_runs):
      # last num_runs examples in S0 are test set
      theta = S0_test[run, :]
      # sample arm features from a Gaussian
      Phi = np.random.randn(K, d)
      Phi /= np.linalg.norm(Phi, axis=-1)[:, np.newaxis]
      # initialize bandit environment
      if reward_model == "linear":
        envs.append(LinBandit(Phi, theta, sigma=sigma))
      elif reward_model == "logistic":
        envs.append(GLMBandit(Phi, K, theta, mean_function=reward_model))
      else:
        raise Exception("Unknown reward model.")

    if reward_model == "linear":
      dpts_cfg = LIN_DIFF_DPTS_PARAMS.get(problem, {
        "num_steps_sgld": num_steps_sgld,
        "step_size_sgld": step_size_sgld,
        "noise_scale": noise_scale,
      })
      dps_eta_val = LIN_DIFF_DPS_PARAMS.get(problem, eta)
      dmap_cfg = LIN_DIFF_DMAP_PARAMS.get(problem, {
        "K_inner": K_inner,
        "dmap_eta": dmap_eta,
      })

      algs = [
        ("LinUCB", {"sigma": sigma}, "green", "--", "LinUCB"),
        ("LinGreedy", {"sigma": sigma}, "greenyellow", "--", "e-greedy"),
        ("LinTS", {"sigma": sigma}, "tab:blue", "--", "LinTS"),
        ("LinDiffTS", {"prior": prior, "theta0": np.zeros(d), "Sigma0": 1e6 * np.eye(d), "sigma": sigma}, "tab:orange", "-", "LinDiffTS"),
        ("LinDiffDPTS", {"prior": prior, "sigma": sigma, **dpts_cfg}, "tab:purple", "-", "DLTS"),
        ("LinDiffDPS", {"prior": prior, "sigma": sigma, "eta": dps_eta_val}, "tab:brown", "-", "DPSG"),
        ("LinDiffDMAP", {"prior": prior, "sigma": sigma, **dmap_cfg}, "tab:pink", "-", "DPSG-MP")]
    elif reward_model == "logistic":
      algs = [
        ("LogTS", {}, "blue", "-", "TS"),
        # ("LogTS", {"theta0": theta0_bar, "Sigma0": Sigma0_bar}, "cyan", "-", "TunedTS"),
        ("LogDiffTS", {"prior": prior, "theta0": np.zeros(d), "Sigma0": 1e6 * np.eye(d)}, "red", "-", "DiffTS"),
        ("LogDiffDPTS", {"prior": prior, "sigma": sigma, "num_steps_sgld": num_steps_sgld, "step_size_sgld": step_size_sgld, "noise_scale": noise_scale}, "black", "-", "DiffDPTS"),
        ("LogDiffDPS", {"prior": prior, "sigma": sigma, "eta": eta}, "purple", "-", "DiffDPS")]

    else:
      raise Exception("Unknown reward model.")

    step = np.arange(1, n + 1)  # for plots
    sube = (step.size // 10) * np.arange(1, 11) - 1

    # simulation
    for alg in algs:
      # all runs for a single algorithm
      alg_class = globals()[alg[0]]
      start = time.time()
      regret, _ = evaluate(alg_class, alg[1], envs, n, printout=False)
      print("%s: %.1f (%.3fs), " % (alg[-1], regret.sum(axis=0).mean(), time.time() - start), end="")

      if super_run > 0:
        old_regret = np.load("Results_linear_20000/%s_%s.npy" % (problem, alg[-1].lower()))
        regret = np.hstack((old_regret, regret))
      np.save("Results_linear_20000/%s_%s.npy" % (problem, alg[-1].lower()), regret)
    print()

  # plots
  # ensure algs exists when skipping simulation
  if 'algs' not in locals():
      algs = [
          ("LinUCB", {}, "green", "--", "LinUCB"),
          ("e-greedy", {}, "greenyellow", "--", "e-greedy"),
          ("LinTS", {}, "tab:blue", "--", "LinTS"),
          ("LinDiffTS", {}, "tab:orange", "-", "LinDiffTS"),
          ("DiffDLTS", {}, "tab:purple", "-", "DLTS"),
          # ("DiffDPSG", {}, "tab:brown", "-", "DPSG"),
          ("DiffDMAPSG", {}, "tab:pink", "-", "DPSG-MP"),
      ]
  if 'step' not in locals():
      step = np.arange(1, n + 1)
      sube = (step.size // 10) * np.arange(1, 11) - 1

  for alg in algs:
    regret = np.load("Results_linear_20000/%s_%s.npy" % (problem, alg[-1].lower()))

    # plot
    cum_regret = regret.cumsum(axis=0)
    style = linestyle2dashes(alg[3])
    if isinstance(style, tuple):
        line, = plt.plot(step, cum_regret.mean(axis=1), color=alg[2], linewidth=2, label=alg[4])
        line.set_dashes(style)
    else:
        plt.plot(step, cum_regret.mean(axis=1), color=alg[2], linestyle=style, linewidth=2, label=alg[4])
    plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
      cum_regret[sube, :].std(axis=1) / np.sqrt(cum_regret.shape[1]),
      fmt="none", ecolor=alg[2])

  plt.title("Problem %s" % problem)
  plt.xlabel("Round n")
  plt.ylabel("Regret")
  # plt.ylim(0, 25)
  plt.legend(loc="upper left", frameon=False)
  plt.savefig("Results_linear_20000/%s_%s.pdf" % (problem, alg[-1].lower()))
  plt.close()