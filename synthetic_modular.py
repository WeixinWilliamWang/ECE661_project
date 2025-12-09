# -------- Std / third-party libs --------
import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# -------- Local project modules --------
from bandit import *  # provides LinBandit, GLMBandit, NonLinearBandit, and all alg classes
from diffusion_posterior import DiffusionPrior

# Torch (used for non-linear branch)
import torch, torch.nn as nn, torch.optim as optim

# helper utils
from utils_synthetic import (
    generate_s0_dataset,
    visualize_prior_scatter,
    visualize_prior_umap,
    linestyle2dashes
)

from tqdm import tqdm


PROBLEMS = ["cross", "rays", "triangles", "swirl", "H", "corners"]
# PROBLEMS = ["cross"]
# PROBLEMS = ["rays"]
# PROBLEMS = ["triangles"]
# PROBLEMS = ["swirl"]
# PROBLEMS = ["H"]
# PROBLEMS = ["corners"]


def prepare_data(problem: str, n_pretrain=10000, num_runs=10):
    """Return S0_train, S0_test and helper stats."""
    S0 = generate_s0_dataset(problem, n_pretrain)  # identical logic as original
    rng = np.random.default_rng(0)
    S0 = rng.permutation(S0)
    S0_train = S0[:-num_runs]
    S0_test = S0[-num_runs:]

    # simple Gaussian stats used by some baselines
    theta0_bar = S0_train.mean(0)
    Sigma0_bar = np.cov(S0_train.T)
    return S0_train, S0_test, dict(theta0_bar=theta0_bar, Sigma0_bar=Sigma0_bar)

######################## LINEAR ########################

def run_linear(problem, S0_train, S0_test, args):
    d = S0_train.shape[1]
    # train diffusion prior on raw θ
    prior = DiffusionPrior(d, T=100, alpha=0.97, reg=1e-4, hidden_size=100)
    prior.train(S0_train)

    # optional visualization
    if args.visualize:
        if d == 2:
            visualize_prior_scatter(S0_train[:1000], prior.sample(1000)[0], f"{args.output_dir}/{problem}_linear_prior_scatter.pdf")
        else:
            visualize_prior_umap(S0_train[:2000], prior.sample(2000)[0], f"{args.output_dir}/{problem}_linear_prior_umap.pdf")

    # build bandit envs
    envs = []
    sigma = args.sigma
    for theta in S0_test:
        Phi = np.random.randn(args.K, d); Phi /= np.linalg.norm(Phi, axis=-1)[:,None]
        envs.append(LinBandit(Phi, theta, sigma=sigma))

    algs = [
        # ("LinGreedy", {"sigma": sigma}, "greenyellow", "-", "e-greedy"),
        ("LinTS", {"sigma": sigma}, "blue", "-", "LinTS"),
        ("LinDiffTS", {"prior": prior, "sigma": sigma, "theta0": np.zeros(d), "Sigma0": 1e6*np.eye(d)}, "red", "-", "LinDiffTS"),
        ("LinDiffDPTS", {"prior": prior, "sigma": sigma, "num_steps_sgld": 1, "step_size_sgld": 0.05, "noise_scale": 0.01}, "green", "-", "LinDLTS"),
        ("LinDiffDPS", {"prior": prior, "sigma": sigma, "eta": 0.005}, "purple", "-", "LinDPSG"),
        ("LinDiffDMAP", {"prior": prior, "sigma": sigma, "K_inner": args.K_inner, "eta": args.dmap_eta}, "brown", "-", "DiffDMAP"),
        ("LinDiffDMAPNew", {"prior": prior, "sigma": sigma, "K_inner": args.K_inner, "eta": args.dmap_eta}, "pink", "-", "DiffDMAPNew")
    ]
    evaluate_and_plot(problem, "linear", algs, envs, args)

###################### LOGISTIC ########################

def run_logistic(problem, S0_train, S0_test, args):
    d = S0_train.shape[1]; sigma=args.sigma
    prior = DiffusionPrior(d, T=100, alpha=0.97, reg=1e-4, hidden_size=100)
    prior.train(S0_train)

    if args.visualize:
        if d == 2:
            visualize_prior_scatter(S0_train[:1000], prior.sample(1000)[0], f"{args.output_dir}/{problem}_logistic_prior_scatter.pdf")
        else:
            visualize_prior_umap(S0_train[:2000], prior.sample(2000)[0], f"{args.output_dir}/{problem}_logistic_prior_umap.pdf")

    envs=[]
    for theta in S0_test:
        Phi=np.random.randn(args.K,d); Phi/=np.linalg.norm(Phi,axis=-1)[:,None]
        envs.append(GLMBandit(Phi,args.K,theta,mean_function="logistic"))

    algs=[
        ("LogTS", {}, "blue", "-", "LogTS"),
        ("LogDiffTS", {"prior": prior, "theta0": np.zeros(d), "Sigma0": 1e6*np.eye(d)}, "red", "-", "LogDiffTS"),
        ("LogDiffDPTS", {"prior": prior, "num_steps_sgld": args.num_steps_sgld, "step_size_sgld": args.step_size_sgld, "noise_scale": args.noise_scale}, "green", "-", "DiffDPTS"),
        ("LogDiffDPS", {"prior": prior, "eta": args.dps_eta}, "purple", "-", "DiffDPS"),
        ("LogDiffDMAP", {"prior": prior, "sigma": sigma, "K_inner": args.K_inner, "eta": args.dmap_eta}, "brown", "-", "DiffDMAP"),
    ]
    evaluate_and_plot(problem,"logistic",algs,envs,args)

#################### NON-LINEAR ########################

def run_non_linear(problem,S0_train,S0_test,args):
    d = S0_train.shape[1]
    sigma = args.sigma
    base_dir = f"{args.output_dir}_{args.type}"
    os.makedirs(base_dir, exist_ok=True)

    # ---------------- Linear weights -> lin_prior (for LinDiffTS baseline) ----------------
    lin_prior_path = os.path.join(base_dir, f"{problem}_diffusion_lin_prior.pkl")
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
    neural_prior_path = os.path.join(base_dir, f"{problem}_neural_prior.pkl")
    if os.path.exists(neural_prior_path):
        neural_prior = joblib.load(neural_prior_path)
    else:
        param_path = os.path.join(base_dir, f"{problem}_mlp_params.npy")
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
                # ----- sample 8 distinct theta vectors -----
                idxs = np.random.choice(len(S0_train), 8, replace=False)
                W = S0_train[idxs]        # (8, d)

                if args.type == "cosine":
                    X = np.random.randn(512, d)
                    y = np.cos(3 * X.dot(theta_vec))
                elif args.type == "quadratic":
                    X = np.random.randn(512, d)
                    y = (X.dot(theta_vec))**2
                elif args.type == "two_layer":
                    X = np.random.randn(512, d)
                    z = X.dot(theta_vec)            # (batch,)
                    a = 2; b = 1
                    y = a * (z*sigmoid(z)) + b
                elif args.type == "two_layer_relu8":
                    X = np.random.randn(512, d)
                    Z = X.dot(W.T)                 # (batch, 8)
                    # ----- sample 4 more theta vectors for second layer weight -----
                    idxs_out = np.random.choice(len(S0_train), 4, replace=False)
                    W2 = S0_train[idxs_out]       # (4, d)
                    w_out = W2.flatten()[:8]      # length 8 weight vector
                    y = np.maximum(Z, 0).dot(w_out)
                elif args.type == "mlp":
                    X = np.random.randn(512, d)
                    h1 = np.maximum(0, theta_vec[0] * X[:, 0] + X[:, 1])
                    h2 = np.maximum(0, X[:, 0] + theta_vec[1] * X[:, 1])
                    y = h1 + h2
                elif args.type == "fm":
                    X = np.random.randn(512, d)
                    linear_term = X.dot(theta_vec)
                    interaction_coeff = theta_vec[0] * theta_vec[1]
                    interaction_term = interaction_coeff * (X[:, 0] * X[:, 1])
                    y = linear_term + interaction_term
                elif args.type == "gam":
                    X = np.random.randn(512, d)
                    f1 = theta_vec[0] * np.sin(np.pi * X[:, 0])
                    f2 = theta_vec[1] * np.cos(np.pi * X[:, 1])
                    y = f1 + f2
                elif args.type == "ga2m":
                    X = np.random.randn(512, d)
                    main_effect = theta_vec[0] * (np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1]))
                    interaction = np.sin(theta_vec[1] * X[:, 0] * X[:, 1] * 5.0)
                    y = main_effect + interaction


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
        joblib.dump(neural_prior, neural_prior_path)

        # ---------- Visualization (first run only) ----------
        if args.visualize:
            out_scatter = os.path.join(base_dir, f"{problem}_nonlin_prior_umap.pdf")
            visualize_prior_umap(mlp_params[:1000], neural_prior.sample(1000)[0], out_scatter)

    # ---------- Build environments ----------
    envs = []
    for theta in S0_test:
        Phi = np.random.randn(args.K, d); Phi /= np.linalg.norm(Phi,axis=-1)[:,None]
        if args.type == "two_layer_relu8":
            # sample 11 additional theta vectors from training set for hidden/output layers
            extra_idxs = np.random.choice(len(S0_train), 11, replace=False)
            theta_stack = np.vstack([theta, S0_train[extra_idxs]])  # (12,d)
            envs.append(NonLinearBandit(Phi, theta_stack, sigma=sigma, type=args.type))
        else:
            envs.append(NonLinearBandit(Phi, theta, sigma=sigma, type=args.type))

    neuralTS = {
        "hidden": args.hidden,
        "nu": args.nu,
        "batch_size": args.batch_size,
        "reduce": args.reduce,
        "reg": args.reg,
        "device": args.device
    }

    neuralUCB = {
        "hidden": args.hidden,
        "nu": args.nu,
        "batch_size": args.batch_size,
        "reduce": args.reduce,
        "reg": args.reg,
        "device": args.device
    }

    algs = [
        ("LinTS", {}, "blue", "-", "TS"),
        ("LinDiffTS", {"prior": lin_prior, "sigma": sigma, "theta0": np.zeros(d), "Sigma0": 1e6*np.eye(d)}, "orange", "-", "DiffTS"),
        ("NeuralTS", neuralTS, "green", "-", "NeuralTS"),
        ("NeuralUCB", neuralUCB, "red", "-", "NeuralUCB"),
        ("NeuralDiffDPTS", {"prior": neural_prior, "sigma": sigma, "num_steps_sgld": args.num_steps_sgld, "step_size_sgld": args.step_size_sgld, "noise_scale": args.noise_scale}, "cyan", "-", "DiffDPTS"),
        ("NeuralDiffDPS", {"prior": neural_prior, "sigma": sigma, "eta": args.dps_eta}, "purple", "-", "DiffDPS"),
        ("NeuralDiffDMAP", {"prior": neural_prior, "sigma": sigma, "K_inner": args.K_inner, "eta": args.dmap_eta}, "brown", "-", "DiffDMAP"),
        ("NeuralDiffDMAPNew", {"prior": neural_prior, "sigma": sigma, "K_inner": args.K_inner, "eta": args.dmap_eta}, "pink", "-", "DiffDMAPNew"),
    ]

    evaluate_and_plot(problem, "nonlinear", algs, envs, args)

    return  # no further filtering needed

##################### COMMON ###########################

def evaluate_and_plot(problem,tag,algs,envs,args):
    # optional single-alg filtering
    if args.alg != "all":
        algs = [a for a in algs if a[0] == args.alg]
        if not algs:
            print(f"[Skip] Algorithm {args.alg} not in list for {problem}-{tag}"); return

    n = args.horizon
    step = np.arange(1, n + 1)
    # select ~10 equally‐spaced points for error bars (skip idx0 for visibility)
    sube = (step.size // 10) * np.arange(1, 11) - 1

    for alg in algs:
        alg_class = globals()[alg[0]]
        start_time = time.time()
        regret, _ = evaluate(alg_class, alg[1], envs, n, printout=False, n_jobs=16)
        elapsed = time.time() - start_time

        cum_regret = regret.cumsum(0)
        mean_cum = cum_regret.mean(1)
        err_cum = cum_regret.std(1) / np.sqrt(cum_regret.shape[1])

        # ---- Print algorithm name and final results to stdout ----
        print(f"{problem}-{tag} | {alg[0]} | Final cumulative regret: {mean_cum[-1]:.2f} | Runtime: {elapsed:.3f}s")

        plt.plot(step, mean_cum, alg[2], label=alg[-1], dashes=linestyle2dashes(alg[3]))
        plt.errorbar(step[sube], mean_cum[sube], err_cum[sube], fmt="none", ecolor=alg[2])

    plt.title(f"{problem} ({tag})")
    plt.xlabel("Round n")
    plt.ylabel("Cumulative Regret")
    plt.legend(frameon=False)
    if args.reward_model == "non-linear":
        out_dir = os.path.join(args.output_dir + '_' + args.type, f"{problem}_{tag}_regret.pdf")
    else:
        out_dir = os.path.join(args.output_dir, f"{problem}_{tag}_regret.pdf")
    plt.tight_layout()
    plt.savefig(out_dir)
    plt.close()

######################   MAIN   ########################

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reward_model", choices=["linear","logistic","non-linear"],default="non-linear")
    parser.add_argument("--sigma",type=float,default=0.1)
    parser.add_argument("--K",type=int,default=100)
    parser.add_argument("--horizon",type=int,default=200)
    # additional hyper-params for nonlinear algorithms
    parser.add_argument("--num_steps_sgld",type=int,default=1)
    parser.add_argument("--step_size_sgld",type=float,default=0.05)
    parser.add_argument("--noise_scale",type=float,default=0.01)
    parser.add_argument("--dps_eta",type=float,default=0.005)
    parser.add_argument("--K_inner",type=int,default=1)
    parser.add_argument("--dmap_eta",type=float,default=0.1)
    # ---- neural algorithm hyper-parameters ----
    parser.add_argument("--hidden",type=int,default=32,help="Hidden layer width for default MLP in Neural algorithms")
    parser.add_argument("--nu",type=float,default=0.01,help="Exploration/confidence parameter for NeuralTS/UCB")
    parser.add_argument("--batch_size",type=int,default=None,help="Mini-batch size (None => full batch)")
    parser.add_argument("--reduce",type=int,default=None,help="Only update NN every <reduce> steps (None => every step)")
    parser.add_argument("--reg",type=float,default=1.0,help="Ridge regularisation weight for design vector")
    parser.add_argument("--device",default="cpu",help="Torch device for Neural algorithms")
    parser.add_argument("--visualize",action="store_true", default=True)
    parser.add_argument("--alg", default="all", help="Run single algorithm class name or 'all'")
    parser.add_argument("--output_dir", default="Results_2000")
    parser.add_argument("--type", default="cosine", choices=["cosine", "quadratic", "two_layer", "two_layer_relu", "two_layer_relu8", "mlp", "fm", "gam", "ga2m"])
    args=parser.parse_args()

    for problem in PROBLEMS:
        S0_train,S0_test,stats=prepare_data(problem)
        if args.reward_model=="linear":
            run_linear(problem,S0_train,S0_test,args)
        elif args.reward_model=="logistic":
            run_logistic(problem,S0_train,S0_test,args)
        elif args.reward_model=="non-linear":
            run_non_linear(problem,S0_train,S0_test,args)
        else:
            raise ValueError("Unknown reward_model")

if __name__=="__main__":
    main()
