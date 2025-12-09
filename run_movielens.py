import argparse
import os
from pathlib import Path
from typing import Union, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_TS import DiffusionPriorMLP
from bandit import (
    LinBandit,
    evaluate,
    LinTS,
    LinDiffTS,
    LinDiffDPTS,
    LinDiffDMAP,
    NeuralTS,
    NeuralUCB,
    NeuralDiffDPTS,
    NeuralDiffDMAP,
)

from train_movielens import load_movielens_data, build_contexts

import wandb

mpl.style.use("classic")
mpl.rcParams["figure.figsize"] = [6, 4]
mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["errorbar.capsize"] = 3
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
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0


def flatten_state_dict(sd: dict[str, torch.Tensor]) -> np.ndarray:
    vec = torch.cat([p.view(-1) for p in sd.values()])
    return vec.cpu().numpy()


def load_mlp_param_matrix(pt_file: Path) -> np.ndarray:
    assert pt_file.exists(), f"MLP params file {pt_file} missing."
    sd_list = torch.load(pt_file, map_location="cpu")
    mat = np.stack([flatten_state_dict(sd) for sd in sd_list]).astype(np.float32)
    return mat


def linestyle2dashes(style: str) -> Union[str, Tuple[int, ...]]:
    mapping = {"-": "-", "--": (6, 2), "-.": (4, 2, 1, 2), ":": (1, 2)}
    return mapping.get(style, style)


# ------- Visualization of prior vs diffusion prior samples (UMAP + PCA) -------
from sklearn.decomposition import PCA  # lightweight import; sklearn already used

def visualize_prior_vs_true(true_mat: np.ndarray, prior_model, out_path: Path, n_vis: int = 1000):
    """UMAP & PCA scatter comparing true parameter samples with diffusion prior samples.

    true_mat: (N, D) numpy array of ground-truth parameter samples used to train prior
    prior_model: DiffusionPriorMLP already fitted; must expose .sample(K) -> (K,D)
    out_path: path of the UMAP figure (PCA figure will use same stem + _pca)
    n_vis: number of points visualised for each set
    """
    import umap  # local import to avoid issues if not installed elsewhere

    if out_path.exists():
        print(f"[Skip] {out_path.name} already exists, skipping UMAP plot.")
        umap_skip = True
    else:
        umap_skip = False

    n_vis = min(n_vis, true_mat.shape[0])
    true_subset = true_mat[:n_vis]
    prior_subset = prior_model.sample(n_vis)[0]

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)
    true_2d = reducer.fit_transform(true_subset)
    prior_2d = reducer.transform(prior_subset)

    if not umap_skip:
        plt.figure(figsize=(6, 6))
        plt.scatter(true_2d[:, 0], true_2d[:, 1], s=6, marker=".", color="b", label="True Prior")
        plt.scatter(prior_2d[:, 0], prior_2d[:, 1], s=6, marker=".", color="grey", label="Diffusion Prior")
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_path)
        plt.close()

    # ------------------ PCA visualization ------------------
    pca_out_path = out_path.parent / f"{out_path.stem}_pca{out_path.suffix}"
    if pca_out_path.exists():
        print(f"[Skip] {pca_out_path.name} already exists, skipping PCA plot.")
        return

    pca = PCA(n_components=2, random_state=0)
    true_pca = pca.fit_transform(true_subset)
    prior_pca = pca.transform(prior_subset)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_pca[:, 0], true_pca[:, 1], s=6, marker=".", color="b", label="True Prior")
    plt.scatter(prior_pca[:, 0], prior_pca[:, 1], s=6, marker=".", color="grey", label="Diffusion Prior")
    plt.tight_layout()
    plt.legend()
    plt.savefig(pca_out_path)
    plt.close()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_movielens_cache(
    cache_path: Path,
    entry_path: Path,
    user_feat_path: Optional[Path],
    movie_feat_path: Optional[Path],
    embed_dim: int,
):
    if cache_path.exists():
        with np.load(cache_path) as npz:
            needed = {"contexts", "labels", "entries", "user_features", "movie_features"}
            if needed.issubset(npz.files):
                print(f"[Cache] loaded Movielens features from {cache_path}")
                return (
                    npz["contexts"],
                    npz["labels"],
                    npz["entries"],
                    npz["user_features"],
                    npz["movie_features"],
                )
            else:
                print(f"[Cache] {cache_path} missing keys; rebuilding.")
    entries, user_features, movie_features = load_movielens_data(
        entry_path,
        user_feat_path=user_feat_path,
        movie_feat_path=movie_feat_path,
        d=embed_dim,
        cache_dir=cache_path.parent,
    )
    contexts, labels = build_contexts(entries, user_features, movie_features)
    np.savez(
        cache_path,
        contexts=contexts.astype(np.float32),
        labels=labels.astype(np.int64),
        entries=entries.astype(np.int64),
        user_features=user_features.astype(np.float32),
        movie_features=movie_features.astype(np.float32),
    )
    return contexts, labels, entries, user_features, movie_features


def build_user_movie_lookup(entries: np.ndarray):
    pos = {}
    neg = {}
    for u, m, r in entries.astype(int):
        # Align with train_movielens.build_contexts: treat r == -1 as the positive label
        if r == -1:
            pos.setdefault(u, []).append(m)
        else:
            neg.setdefault(u, []).append(m)
    return pos, neg


class MovielensBandit(LinBandit):
    """Bandit that builds arms for a single user with 1 positive and K-1 negatives."""

    def __init__(
        self,
        user_id: int,
        user_features: np.ndarray,
        movie_features: np.ndarray,
        pos_dict: dict,
        neg_dict: dict,
        sigma: float = 0.0,
        K: int = 10,
    ):
        self.user_id = int(user_id)
        self.user_features = user_features.astype(np.float32)
        self.movie_features = movie_features.astype(np.float32)
        self.pos_dict = pos_dict
        self.neg_dict = neg_dict
        self.K = K
        self.d_user = self.user_features.shape[1]
        self.d_movie = self.movie_features.shape[1]
        self.d = self.d_user + self.d_movie
        super().__init__(np.zeros((self.K, self.d)), np.zeros(self.d), sigma=sigma)

    def randomize(self):
        rng = np.random.default_rng()
        positives = self.pos_dict.get(self.user_id, [])
        negatives = self.neg_dict.get(self.user_id, [])

        arms_movies = []
        rewards = []
        if positives:
            pos_movie = int(rng.choice(positives))
            arms_movies.append(pos_movie)
            rewards.append(1.0)
        else:
            # No positive movies observed; fall back to a random negative
            fallback = int(rng.choice(negatives)) if negatives else int(rng.integers(len(self.movie_features)))
            arms_movies.append(fallback)
            rewards.append(0.0)

        need = self.K - len(arms_movies)
        pool = [m for m in negatives if m not in arms_movies]
        if len(pool) < need:
            pool = negatives if negatives else list(range(len(self.movie_features)))
            extra = rng.choice(pool, size=need, replace=len(pool) < need)
        else:
            extra = rng.choice(pool, size=need, replace=False)
        arms_movies.extend(extra.tolist())
        rewards.extend([0.0] * len(extra))

        order = rng.permutation(self.K)
        arms_movies = np.array(arms_movies)[order]
        rewards = np.array(rewards, dtype=np.float32)[order]

        ctx = []
        user_vec = self.user_features[self.user_id]
        for mid in arms_movies:
            movie_vec = self.movie_features[int(mid)]
            ctx.append(np.concatenate([user_vec, movie_vec], axis=0))
        self.X = np.stack(ctx, axis=0).astype(np.float32)
        self.mu = rewards
        self.rt = self.mu + self.sigma * np.random.randn(self.K) if self.sigma > 0 else self.mu.copy()
        self.best_arm = int(np.argmax(self.mu))


def main(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lin_theta_path = Path(args.linear_thetas)
    mlp_param_path = Path(args.mlp_params)
    if not lin_theta_path.exists() or not mlp_param_path.exists():
        raise FileNotFoundError("Run train_movielens.py first to generate priors.")

    contexts, labels, entries, user_features, movie_features = get_movielens_cache(
        Path(args.dataset_npz),
        Path(args.entry),
        Path(args.user_features) if args.user_features else None,
        Path(args.movie_features) if args.movie_features else None,
        args.embed_dim,
    )
    d_feat = contexts.shape[1]
    num_users = user_features.shape[0]
    train_user_cutoff = int(num_users * args.train_frac_users)
    print(f"Using contexts dim={d_feat}, train users < {train_user_cutoff}")

    theta_samples = np.load(lin_theta_path)
    perm = np.random.permutation(theta_samples.shape[0])
    n_train_lin = int(theta_samples.shape[0] * args.prior_train_frac)
    theta_train = theta_samples[perm[:n_train_lin]]
    theta_test = theta_samples[perm[n_train_lin:]]

    lin_prior_path = output_dir / "movielens_diffusion_prior_linear_sd.pt"
    lin_prior = DiffusionPriorMLP(
        d=d_feat,
        T=args.T,
        alpha=args.alpha,
        hidden=args.hidden_size,
        device=args.device,
        schedule=args.schedule,
        arch="mlp",
    )
    if lin_prior_path.exists():
        print(f"[Cache] load linear prior from {lin_prior_path}")
        lin_prior.load_state_dict(torch.load(lin_prior_path, map_location=args.device))
    else:
        lin_prior.train(theta_train, epochs=args.epochs, batch=args.batch, lr=args.lr, log_prefix="linear_prior")
        torch.save(lin_prior.state_dict(), lin_prior_path)

    # -------- visualize linear theta prior vs diffusion prior --------
    visualize_prior_vs_true(theta_train, lin_prior, output_dir / "linear_theta_prior_umap.pdf")

    neural_mat = load_mlp_param_matrix(mlp_param_path)
    P = neural_mat.shape[1]
    perm2 = np.random.permutation(neural_mat.shape[0])
    n_train_neural = int(neural_mat.shape[0] * args.prior_train_frac)
    neural_train = neural_mat[perm2[:n_train_neural]]
    neural_test = neural_mat[perm2[n_train_neural:]]

    neural_prior_path = output_dir / "movielens_diffusion_prior_mlp_sd.pt"
    neural_prior = DiffusionPriorMLP(
        d=P,
        T=args.T,
        alpha=args.alpha,
        hidden=args.hidden_size,
        device=args.device,
        schedule=args.schedule,
        arch="transformer",
    )
    if neural_prior_path.exists():
        print(f"[Cache] load neural prior from {neural_prior_path}")
        neural_prior.load_state_dict(torch.load(neural_prior_path, map_location=args.device))
    else:
        neural_prior.train(neural_train, epochs=args.epochs, batch=args.batch, lr=args.lr, log_prefix="mlp_prior")
        torch.save(neural_prior.state_dict(), neural_prior_path)

    # -------- visualize MLP param prior vs diffusion prior --------
    visualize_prior_vs_true(neural_train, neural_prior, output_dir / "mlp_param_prior_umap.pdf")

    visualize_prior_vs_true(theta_train, lin_prior, output_dir / "movielens_linear_prior_vs_true.pdf")
    visualize_prior_vs_true(neural_train, neural_prior, output_dir / "movielens_mlp_prior_vs_true.pdf")

    pos_dict, neg_dict = build_user_movie_lookup(entries)
    candidate_users = [u for u in range(num_users) if len(pos_dict.get(u, [])) > 0]
    envs = [
        MovielensBandit(
            user_id=int(np.random.choice(candidate_users)),
            user_features=user_features,
            movie_features=movie_features,
            pos_dict=pos_dict,
            neg_dict=neg_dict,
            sigma=args.sigma,
            K=args.num_arms,
        )
        for _ in range(args.runs)
    ]

    neural_common = {
        "hidden": args.hidden_neural,
        "nu": args.nu,
        "batch_size": args.batch_size,
        "reduce": None,
        "reg": args.reg,
        "device": args.device,
    }
    neural_ts_params = neural_common.copy()
    neural_ucb_params = neural_common.copy()

    full_alg_list = {
        "LinTS": ("LinTS", {"sigma": args.sigma}, "tab:blue", "--", "TS"),
        "LinDiffTS": (
            "LinDiffTS",
            {
                "prior": lin_prior,
                "sigma": args.sigma,
                "theta0": np.zeros(d_feat),
                "Sigma0": 1e6 * np.eye(d_feat),
            },
            "tab:orange",
            "-",
            "DiffTS",
        ),
        "NeuralTS": ("NeuralTS", neural_ts_params, "tab:green", "--", "NeuralTS"),
        "NeuralUCB": ("NeuralUCB", neural_ucb_params, "tab:red", "-.", "NeuralUCB"),
        "NeuralDiffDPTS": (
            "NeuralDiffDPTS",
            {
                "prior": neural_prior,
                "sigma": args.sigma,
                "num_steps_sgld": args.num_steps_sgld,
                "step_size_sgld": args.step_size_sgld,
                "noise_scale": args.noise_scale,
            },
            "tab:purple",
            "-",
            "DLTS",
        ),
        "NeuralDiffDMAP": (
            "NeuralDiffDMAP",
            {
                "prior": neural_prior,
                "sigma": args.sigma,
                "K_inner": args.K_inner,
                "eta": args.dmap_eta,
            },
            "tab:olive",
            "--",
            "DMAP",
        ),
    }

    requested = args.alg.strip().lower()
    available_algs = list(full_alg_list.keys())
    if requested == "all":
        selected = available_algs
    else:
        req = [a.strip() for a in args.alg.split(",") if a.strip()]
        bad = sorted(set(req) - set(available_algs))
        if bad:
            raise ValueError(f"Unknown algorithms {bad}. Available: {available_algs}")
        selected = req

    algs = [full_alg_list[name] for name in selected]
    step = np.arange(1, args.horizon + 1)
    sube = np.linspace(0, args.horizon - 1, num=10, dtype=int)

    for alg in algs:
        alg_class = globals()[alg[0]]
        print(f"[RUN] {alg[0]} ({alg[4]})", flush=True)
        regret, _ = evaluate(alg_class, alg[1], envs, args.horizon, printout=True)
        print(f"[DONE] {alg[0]} ({alg[4]})", flush=True)
        cum_reg = regret.cumsum(axis=0)
        alg_name = alg[0]
        # Build filename suffix from primitive params (int, float, str, bool)
        simple_params = {k: v for k, v in alg[1].items() if isinstance(v, (int, float, str, bool))}
        param_suffix = "_".join([f"{k}{simple_params[k]}" for k in sorted(simple_params)])
        result_path = (
            results_dir / f"{alg_name}_{param_suffix}_results.npz" if param_suffix else results_dir / f"{alg_name}_results.npz"
        )
        np.savez_compressed(
            result_path,
            regret=regret,
            cumulative_regret=cum_reg,
            step=step,
            algorithm=np.array(alg[0]),
            display_label=np.array(alg[4]),
            runs=np.array(args.runs, dtype=np.int32),
            horizon=np.array(args.horizon, dtype=np.int32),
            sigma=np.array(args.sigma, dtype=np.float32),
        )
        style = linestyle2dashes(alg[3])
        line_kwargs = dict(color=alg[2], label=alg[4], linewidth=2)
        if isinstance(style, tuple):
            line, = plt.plot(step, cum_reg.mean(axis=1), **line_kwargs)
            line.set_dashes(style)
        else:
            plt.plot(step, cum_reg.mean(axis=1), linestyle=style, **line_kwargs)
        plt.errorbar(
            step[sube],
            cum_reg[sube].mean(axis=1),
            cum_reg[sube].std(axis=1) / np.sqrt(cum_reg.shape[1]),
            fmt="none",
            ecolor=alg[2],
        )
        print(f"{alg[4]} final cumulative regret: {cum_reg.mean(axis=1)[-1]:.2f}")

    plt.title("Movielens Bandit")
    plt.xlabel("Round n")
    plt.ylabel("Cumulative Regret")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "movielens_bandit_regret.pdf")
    plt.savefig(plot_path)
    plt.close()
    print("Plot saved to", plot_path)
    print("Finished Movielens bandit experiment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movielens contextual bandit (1 positive + 9 negative arms)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--entry", type=str, default="Movielens/movie_2000users_10000items_entry.npy")
    parser.add_argument("--user_features", type=str, default="Movielens/movie_2000users_10000items_features.npy")
    parser.add_argument("--movie_features", type=str, default=None)
    parser.add_argument("--dataset_npz", type=str, default="Results_movielens_transformer/movielens_features_labels.npz")
    parser.add_argument("--linear_thetas", type=str, default="Results_movielens_transformer/movielens_linear_thetas.npy")
    parser.add_argument("--mlp_params", type=str, default="Results_movielens_transformer/movielens_mlp_state_dicts.pt")
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--output_dir", default="Results_movielens_transformer")
    parser.add_argument("--train_frac_users", type=float, default=1.0, help="users < cutoff used for training priors/classifier")
    parser.add_argument("--prior_train_frac", type=float, default=1.0, help="fraction of parameter samples used to train diffusion priors")
    parser.add_argument("--num_arms", type=int, default=10)

    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--schedule", type=str, choices=["const", "cosine", "linear"], default="linear")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--hidden_neural", type=int, default=32)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--reg", type=float, default=0.1)

    parser.add_argument("--num_steps_sgld", type=int, default=1)
    parser.add_argument("--step_size_sgld", type=float, default=0.05)
    parser.add_argument("--noise_scale", type=float, default=0.01)
    parser.add_argument("--dmap_eta", type=float, default=0.1)
    parser.add_argument("--K_inner", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alg", type=str, default="all")

    args = parser.parse_args()
    wandb.init(project="movielens-bandit", name=args.alg, config=args)
    main(args)
    wandb.finish()
