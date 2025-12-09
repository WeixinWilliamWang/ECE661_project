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

import torch
import torch.nn as nn
import torch.optim as optim

import umap.umap_ as umap

import os

RESULTS_DIR = "Results_non_linear_mlp_mlp"  # <== new output directory
from utils_synthetic import visualize_prior_umap
import argparse

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
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["legend.fontsize"] = "medium"



mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0

import platform
print("python %s" % platform.python_version())
print("matplotlib %s" % mpl.__version__)
print("%d joblib CPUs" % joblib.cpu_count())

import warnings
import gc
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
parser.add_argument("--algs", type=str, default="all", help="Comma separated list of algorithms to run, or 'all'")
args = parser.parse_args()





d = 2  # number of features
K = 100  # number of arms
n = 200  # horizon
num_runs = 16  # number of random runs per super run
num_super_runs = 8  # priors are re-estimated
reward_model = "non_linear"  # reward model
sigma = 1.0  # reward noise
n_pretrain = 10000  # number of samples for estimating the prior

if args.problem.lower() == "all":
  problems = ["cross", "rays", "triangles", "swirl", "H", "corners"]
else:
  problems = [args.problem]

if args.problem.lower() == "cross":
  neuralTS = {
    "hidden": 32,
    "nu": 0.01,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.01,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.001,
    "device": "cpu"
  }
  num_steps_sgld = 1
  step_size_sgld = 0.05
  noise_scale = 0.1
  eta = 0.001
  K_inner = 1
  dmap_eta = 0.05
elif args.problem.lower() == "rays":
  neuralTS = {
    "hidden": 32,
    "nu": 0.01,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.001,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 0.1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.01,
    "device": "cpu"
  }
  num_steps_sgld = 10
  step_size_sgld = 0.01
  noise_scale = 0.005
  K_inner = 10
  dmap_eta = 0.001
  eta = 0.001
elif args.problem.lower() == "triangles":
  neuralTS = {
    "hidden": 32,
    "nu": 0.1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.001,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.01,
    "device": "cpu"
  }
  num_steps_sgld = 10
  step_size_sgld = 0.05
  noise_scale = 0.01
  K_inner = 10
  dmap_eta = 0.01
  eta = 0.001
elif args.problem.lower() == "swirl":
  neuralTS = {
    "hidden": 32,
    "nu": 1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.001,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.1,
    "device": "cpu"
  }
  num_steps_sgld = 10
  step_size_sgld = 0.005
  noise_scale = 0.05
  K_inner = 1
  dmap_eta = 0.05
  eta = 0.001
elif args.problem.lower() == "h":
  neuralTS = {
    "hidden": 32,
    "nu": 0.1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.01,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 0.001,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.1,
    "device": "cpu"
  }
  num_steps_sgld = 10
  step_size_sgld = 0.005
  noise_scale = 0.01
  K_inner = 1
  dmap_eta = 0.005
  eta = 0.001
elif args.problem.lower() == "corners":
  neuralTS = {
    "hidden": 32,
    "nu": 0.1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.1,
    "device": "cpu"
  }
  neuralUCB = {
    "hidden": 32,
    "nu": 1,
    "batch_size": 128,
    "reduce": True,
    "reg": 0.001,
    "device": "cpu"
  }
  num_steps_sgld = 10
  step_size_sgld = 0.05
  noise_scale = 0.01
  K_inner = 10
  dmap_eta = 0.01
  eta = 0.001
else:
  raise Exception("Unknown problem.")





for problem in problems:
  result_tags = ["TS","DiffTS","NeuralTS","NeuralUCB","DLTS","DPSG","DPSG-MP"]
  if all(os.path.exists(f"{RESULTS_DIR}/{problem}_{tag.lower()}.npy") for tag in result_tags):
    print(f"[Info] Results for problem {problem} already exist, skip simulation.")
    num_super_runs_local = 0
  else:
    num_super_runs_local = num_super_runs

  agg_final = {}  # <== aggregate final regret over super runs

  for super_run in range(num_super_runs_local):
    if super_run == 0:
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
      d = S0_train.shape[1]

    # ---------------- Linear weights -> lin_prior (for LinDiffTS baseline) ----------------
    lin_prior_path = f"{RESULTS_DIR}/{problem}_diffusion_lin_prior.pkl"
    import joblib
    if os.path.exists(lin_prior_path):
        lin_prior = joblib.load(lin_prior_path)
    else:
        # Build linear parameter dataset
        def fit_linear(theta_vec, n_samples=512):
            X = np.random.randn(n_samples, d)
            y = np.cos(3 * X.dot(theta_vec))
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            return w

        lin_params = np.stack([fit_linear(th) for th in S0_train], axis=0)
        lin_prior = DiffusionPrior(d, T=100, alpha=0.97, reg=1e-4, hidden_size=100)
        lin_prior.train(lin_params)
        joblib.dump(lin_prior, lin_prior_path)

     # ---------------- MLP params -> neural diffusion prior ----------------
    diffusion_prior_path = f"{RESULTS_DIR}/{problem}_neural_prior.pkl"
    if os.path.exists(diffusion_prior_path):
        neural_prior = joblib.load(diffusion_prior_path)
        param_path = f"{RESULTS_DIR}/{problem}_mlp_params.npy"
        mlp_params = np.load(param_path)
    else:
      param_path = f"{RESULTS_DIR}/{problem}_mlp_params.npy"
      if os.path.exists(param_path):
          mlp_params = np.load(param_path)
      else:
          # Small 2-layer MLP definition
          class SmallMLP(nn.Module):
              def __init__(self, hidden=32):
                  super().__init__()
                  self.fc1 = nn.Linear(d, hidden)
                  self.act = nn.Tanh()
                  self.fc2 = nn.Linear(hidden, 1)

              def forward(self,x):
                  return self.fc2(self.act(self.fc1(x)))

          # ---- star-shaped fine-tuning: first model random, others start from it ----
          base_sd = None
          mlp_param_list = []

          for theta_vec in tqdm(S0_train, desc=f"{problem} MLP (star)"):
              X = np.random.randn(512, d)
              y = np.cos(3 * X.dot(theta_vec))

              net = SmallMLP().to('cpu')
              if base_sd is not None:
                  net.load_state_dict(base_sd)

              opt = optim.Adam(net.parameters(), lr=1e-2)
              X_t = torch.from_numpy(X).float(); y_t = torch.from_numpy(y).float().unsqueeze(1)
              for _ in range(100):
                  opt.zero_grad(); loss = nn.MSELoss()(net(X_t), y_t); loss.backward(); opt.step()

              with torch.no_grad():
                  params = torch.cat([p.flatten() for p in net.parameters()]).cpu().numpy()
              mlp_param_list.append(params)

              # save first trained state dict as base
              if base_sd is None:
                  base_sd = {k: v.clone() for k, v in net.state_dict().items()}

          mlp_params = np.stack(mlp_param_list, 0)
          np.save(param_path, mlp_params)

      d_mlp = mlp_params.shape[1]
      neural_prior = DiffusionPrior(d_mlp,T=100,alpha=0.99,reg=1e-4,hidden_size=256)
      neural_prior.train(mlp_params)
      joblib.dump(neural_prior, diffusion_prior_path)

    out_scatter = f"{RESULTS_DIR}/{problem}_nonlin_prior_umap.pdf"

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)
    real_2d = reducer.fit_transform(mlp_params[:1000])
    prior_2d = reducer.transform(neural_prior.sample(1000)[0])
    plt.figure(figsize=(6, 6))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], s=6, marker=".", color="b", label="True Prior")
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], s=6, marker=".", color="grey", label="Diffusion Prior")
    plt.ylabel("True and diffusion priors")
    plt.tight_layout()
    plt.legend()      
    plt.savefig("Results_non_linear_mlp_mlp/%s_diffusion_prior.pdf" % problem)
    plt.close()

    del mlp_params
    gc.collect()

    # ---------- Build environments ----------
    envs = []
    for run in range(num_runs):
        theta = S0_test[run, :] 
        Phi = np.random.randn(K, d); Phi /= np.linalg.norm(Phi,axis=-1)[:,None]
        envs.append(NonLinearBandit(Phi, theta, sigma=sigma, type='mlp'))

    algs = [
        ("LinTS", {}, "tab:blue", "--", "TS"),
        ("LinDiffTS", {"prior": lin_prior, "sigma": sigma, "theta0": np.zeros(d), "Sigma0": 1e6*np.eye(d)}, "tab:orange", "-", "DiffTS"),
        ("NeuralTS", neuralTS, "tab:green", "--", "NeuralTS"),
        ("NeuralUCB", neuralUCB, "tab:red", "--", "NeuralUCB"),
        ("NeuralDiffDPTS", {"prior": neural_prior, "sigma": sigma, "num_steps_sgld": num_steps_sgld, "step_size_sgld": step_size_sgld, "noise_scale": noise_scale}, "tab:purple", "-", "DLTS"),
        ("NeuralDiffDPS", {"prior": neural_prior, "sigma": sigma, "eta": eta}, "tab:brown", "-", "DPSG"),
        ("NeuralDiffDMAP", {"prior": neural_prior, "sigma": sigma, "K_inner": K_inner, "eta": dmap_eta}, "tab:pink", "-", "DPSG-MP")
    ]
    # filter if user specified subset
    if args.algs.lower() != "all":
        selected = [s.strip() for s in args.algs.split(',')]
        algs = [alg for alg in algs if (alg[4] in selected or alg[0] in selected)]
    step = np.arange(1, n + 1)  # for plots
    sube = (step.size // 10) * np.arange(1, 11) - 1

    # simulation
    for alg in algs:
      # all runs for a single algorithm
      alg_class = globals()[alg[0]]
      start = time.time()
      regret, _ = evaluate(alg_class, alg[1], envs, n, printout=False)
      print("%s: %.1f (%.3fs), " % (alg[-1], regret.sum(axis=0).mean(), time.time() - start), end="")

      if super_run > 0 and os.path.exists(f"{RESULTS_DIR}/{problem}_{alg[-1].lower()}.npy"):
          old_regret = np.load(f"{RESULTS_DIR}/{problem}_{alg[-1].lower()}.npy")
          regret = np.hstack((old_regret, regret))
      np.save(f"{RESULTS_DIR}/{problem}_{alg[-1].lower()}.npy", regret)

      cum_regret = regret.cumsum(axis=0)
      final_mean = cum_regret[-1].mean()
      final_std = cum_regret[-1].std()
      agg_final.setdefault(alg[4], []).append(final_mean)
      print(f"{alg[-1]}: {final_mean:.2f}±{final_std:.2f} ({time.time()-start:.3f}s)")
    print()


    # # bandit environments
    # envs = []
    # for run in range(num_runs):
    #   # last num_runs examples in S0 are test set
    #   theta = S0_test[run, :]
    #   # sample arm features from a Gaussian
    #   Phi = np.random.randn(K, d)
    #   Phi /= np.linalg.norm(Phi, axis=-1)[:, np.newaxis]
    #   # initialize bandit environment
    #   if reward_model == "linear":
    #     envs.append(LinBandit(Phi, theta, sigma=sigma))
    #   elif reward_model == "logistic":
    #     envs.append(GLMBandit(Phi, K, theta, mean_function=reward_model))
    #   else:
    #     raise Exception("Unknown reward model.")

    # if reward_model == "linear":
    #   algs = [
    #     # ("LinUCB", {"sigma": sigma}, "green", "-", "LinUCB"),
    #     # ("LinGreedy", {"sigma": sigma}, "greenyellow", "-", "e-greedy"),
    #     ("LinTS", {"sigma": sigma}, "blue", "-", "LinTS"),
    #     # ("LinTS", {"theta0": theta0_bar, "Sigma0": Sigma0_bar, "sigma": sigma}, "cyan", "-", "TunedTS"),
    #     # ("MixTS", {"p0": p0_gm, "theta0": theta0_gm, "Sigma0": Sigma0_gm, "sigma": sigma}, "orange", "-", "MixTS"),
    #     # ("LinDiffTSChung", {"prior": prior, "sigma": sigma}, "gray", "-", "DiffTSChung"),
    #     ("LinDiffTS", {"prior": prior, "theta0": np.zeros(d), "Sigma0": 1e6 * np.eye(d), "sigma": sigma}, "red", "-", "DiffTS"),
    #     ("LinDiffDPTS", {"prior": prior, "sigma": sigma, "num_steps_sgld": num_steps_sgld, "step_size_sgld": step_size_sgld, "noise_scale": noise_scale}, "black", "-", "DiffDPTS"),
    #     ("LinDiffDPS", {"prior": prior, "sigma": sigma, "eta": eta}, "purple", "-", "DiffDPS")]
    # elif reward_model == "logistic":
    #   algs = [
    #     ("LogTS", {}, "blue", "-", "TS"),
    #     # ("LogTS", {"theta0": theta0_bar, "Sigma0": Sigma0_bar}, "cyan", "-", "TunedTS"),
    #     ("LogDiffTS", {"prior": prior, "theta0": np.zeros(d), "Sigma0": 1e6 * np.eye(d)}, "red", "-", "DiffTS"),
    #     ("LogDiffDPTS", {"prior": prior, "sigma": sigma, "num_steps_sgld": num_steps_sgld, "step_size_sgld": step_size_sgld, "noise_scale": noise_scale}, "black", "-", "DiffDPTS"),
    #     ("LogDiffDPS", {"prior": prior, "sigma": sigma, "eta": eta}, "purple", "-", "DiffDPS")]

    # else:
    #   raise Exception("Unknown reward model.")

    # step = np.arange(1, n + 1)  # for plots
    # sube = (step.size // 10) * np.arange(1, 11) - 1

    # # simulation
    # for alg in algs:
    #   # all runs for a single algorithm
    #   alg_class = globals()[alg[0]]
    #   start = time.time()
    #   regret, _ = evaluate(alg_class, alg[1], envs, n, printout=False)
    #   print("%s: %.1f (%.3fs), " % (alg[-1], regret.sum(axis=0).mean(), time.time() - start), end="")

    #   if super_run > 0:
    #     old_regret = np.load("Results_logistic/%s_%s.npy" % (problem, alg[-1].lower()))
    #     regret = np.hstack((old_regret, regret))
    #   np.save("Results_logistic/%s_%s.npy" % (problem, alg[-1].lower()), regret)
    # print()

  # ---- after all super runs, print aggregate statistics ----
  if agg_final:
      print(f"\n[Summary over {num_super_runs_local} super runs | Problem {problem}]")
      for name, vals in agg_final.items():
          vals = np.asarray(vals)
          print(f"  {name}: mean={vals.mean():.2f} ± {vals.std():.2f} (SD) over {len(vals)} runs")

  if 'algs' not in locals():
      algs = [
          ("LinTS", {}, "tab:blue", "--", "LinTS"),
          ("LinDiffTS", {}, "tab:orange", "-", "DiffTS"),
          ("NeuralTS", {}, "tab:green", "--", "NeuralTS"),
          ("NeuralUCB", {}, "tab:red", "--", "NeuralUCB"),
          ("NeuralDiffDPTS", {}, "tab:purple", "-", "DLTS"),
          ("NeuralDiffDPS", {}, "tab:brown", "-", "DPSG"),
          ("NeuralDiffDMAP", {}, "tab:pink", "-", "DPSG-MP"),
      ]

  if 'step' not in locals():
      step = np.arange(1, n + 1)
      sube = (step.size // 10) * np.arange(1, 11) - 1

  # plots
  for alg in algs:
    regret = np.load(f"{RESULTS_DIR}/{problem}_{alg[-1].lower()}.npy")

    # plot
    cum_regret = regret.cumsum(axis=0)
    style = linestyle2dashes(alg[3])
    if isinstance(style, tuple):
        line, = plt.plot(step, cum_regret.mean(axis=1), color=alg[2], label=alg[4], linewidth=2)
        line.set_dashes(style)
    else:
        plt.plot(step, cum_regret.mean(axis=1), color=alg[2], linestyle=style, label=alg[4], linewidth=2)
    plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
      cum_regret[sube, :].std(axis=1) / np.sqrt(cum_regret.shape[1]),
      fmt="none", ecolor=alg[2])

  plt.title("Problem %s" % problem)
  plt.xlabel("Round n")
  plt.ylabel("Regret")
  plt.legend(loc="upper left", frameon=False)
  plt.savefig(f"{RESULTS_DIR}/{problem}_{alg[-1].lower()}.pdf")
  plt.close()