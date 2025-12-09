import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_svd_features(matrix: np.ndarray, d: int = 10) -> np.ndarray:
    """SVD-based embedding used in process_movielens.ipynb."""
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    u = u[:, : d - 1]
    norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    u = u / norms
    feats = np.concatenate([u, np.ones((u.shape[0], 1))], axis=1) / np.sqrt(2)
    return feats.astype(np.float32)


def load_movielens_data(
    entry_path: Path,
    user_feat_path: Optional[Path] = None,
    movie_feat_path: Optional[Path] = None,
    d: int = 10,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load triplets and (user, movie) embeddings; compute embeddings if missing."""
    entries = np.load(entry_path)
    num_users = int(entries[:, 0].max()) + 1
    num_movies = int(entries[:, 1].max()) + 1

    user_features = None
    movie_features = None

    if user_feat_path and user_feat_path.exists():
        uf = np.load(user_feat_path)
        if uf.shape[0] == num_users:
            user_features = uf.astype(np.float32)
        else:
            print(f"[Warn] user feature shape {uf.shape} mismatches num_users={num_users}, recomputing.")

    if movie_feat_path and movie_feat_path.exists():
        mf = np.load(movie_feat_path)
        if mf.shape[0] == num_movies:
            movie_features = mf.astype(np.float32)
        else:
            print(f"[Warn] movie feature shape {mf.shape} mismatches num_movies={num_movies}, recomputing.")

    if user_features is None and movie_features is None and user_feat_path and user_feat_path.exists():
        # Check if single file stores both user and movie embeddings concatenated
        all_feats = np.load(user_feat_path)
        if all_feats.shape[0] == num_users + num_movies:
            user_features = all_feats[:num_users].astype(np.float32)
            movie_features = all_feats[num_users:].astype(np.float32)

    if user_features is None or movie_features is None:
        matrix = np.zeros((num_users, num_movies), dtype=np.float32)
        matrix[entries[:, 0].astype(int), entries[:, 1].astype(int)] = entries[:, 2].astype(np.float32)
        if user_features is None:
            user_features = compute_svd_features(matrix, d=d)
        if movie_features is None:
            movie_features = compute_svd_features(matrix.T, d=d)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_dir / "user_embeddings.npy", user_features)
            np.save(cache_dir / "movie_embeddings.npy", movie_features)

    return entries, user_features, movie_features


def build_contexts(entries: np.ndarray, user_features: np.ndarray, movie_features: np.ndarray, task: str = "classification") -> Tuple[np.ndarray, np.ndarray]:
    """Return concatenated contexts and labels based on task."""
    user_ctx = user_features[entries[:, 0].astype(int)]
    movie_ctx = movie_features[entries[:, 1].astype(int)]
    contexts = np.concatenate([user_ctx, movie_ctx], axis=1).astype(np.float32)
    # Both tasks derive binary labels: 1 if rating == -1, else 0
    labels = (entries[:, 2] == -1).astype(np.float32)
    return contexts, labels


def split_by_user(entries: np.ndarray, contexts: np.ndarray, labels: np.ndarray, train_frac: float = 0.9):
    num_users = int(entries[:, 0].max()) + 1
    cutoff = int(num_users * train_frac)
    mask_train = entries[:, 0] < cutoff
    mask_test = ~mask_train
    train_data = (contexts[mask_train], labels[mask_train])
    test_data = (contexts[mask_test], labels[mask_test])
    return train_data, test_data, cutoff


class InteractionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def sample_linear_thetas(
    contexts: np.ndarray,
    labels: np.ndarray,
    *,
    n_samples: int = 1000,
    subset_size: int = 256,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Closed-form linear fits over random balanced subsets."""
    d = contexts.shape[1]
    thetas = np.zeros((n_samples, d), dtype=np.float32)
    pos_idx = np.where(labels > 0)[0]
    neg_idx = np.where(labels <= 0)[0]

    for i in range(n_samples):
        use_replace_pos = len(pos_idx) < subset_size // 2
        use_replace_neg = len(neg_idx) < subset_size // 2
        idx_p = np.random.choice(pos_idx, subset_size // 2, replace=use_replace_pos)
        idx_n = np.random.choice(neg_idx, subset_size // 2, replace=use_replace_neg)
        idx = np.concatenate([idx_p, idx_n])
        X = contexts[idx]
        y = np.concatenate([np.ones(len(idx_p)), -np.ones(len(idx_n))])
        XtX = X.T @ X + ridge * np.eye(d, dtype=np.float32)
        try:
            theta = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            # Fall back to least-squares solution if XtX is singular
            theta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        thetas[i] = theta.astype(np.float32)
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"[Linear {i+1}/{n_samples}]")
    return thetas


def main():
    parser = argparse.ArgumentParser(description="Train Movielens embeddings + priors for bandits.")
    parser.add_argument("--entry", type=Path, default=Path("Movielens/movie_2000users_10000items_entry.npy"))
    parser.add_argument("--user_features", type=Path, default=Path("Movielens/movie_2000users_10000items_features.npy"))
    parser.add_argument("--movie_features", type=Path, default=Path("Movielens/movie_2000users_10000items_features.npy"))
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="regression", help="learning objective")
    parser.add_argument("--linear_out", type=Path, default=Path("Results_movielens_transformer/movielens_linear_thetas.npy"))
    parser.add_argument("--mlp_out", type=Path, default=Path("Results_movielens_transformer/movielens_mlp_state_dicts.pt"))
    parser.add_argument("--dataset_out", type=Path, default=Path("Results_movielens_transformer/movielens_features_labels.npz"))
    parser.add_argument("--n_linear", type=int, default=2000)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(0)
    args.linear_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)

    entries, user_features, movie_features = load_movielens_data(
        args.entry,
        user_feat_path=args.user_features,
        movie_feat_path=args.movie_features,
        d=args.embed_dim,
        cache_dir=args.dataset_out.parent,
    )
    contexts, labels = build_contexts(entries, user_features, movie_features, task=args.task)
    (ctx_train, y_train), (ctx_test, y_test), cutoff = split_by_user(entries, contexts, labels, train_frac=args.train_frac)
    print(f"Loaded interactions: {len(entries)}  train_users<{cutoff} -> train_samples={len(ctx_train)}, test_samples={len(ctx_test)}")

    # ---- Per-user MLP training ----
    if not args.mlp_out.exists():
        num_users = int(entries[:, 0].max()) + 1
        per_user_states = []
        base_sd = None  # weights of the first trained user model
        for uid in range(num_users):
            idx_all = np.where(entries[:, 0] == uid)[0]
            if len(idx_all) == 0:
                per_user_states.append(None)
                continue

            # ---- per-user train / test split ----
            rng = np.random.default_rng(seed=uid)
            perm = rng.permutation(idx_all)
            n_test_u = max(1, int(len(perm) * (1 - args.train_frac)))
            test_idx_u = perm[:n_test_u]
            train_idx_u = perm[n_test_u:]

            X_train_u = torch.from_numpy(contexts[train_idx_u]).float()
            y_train_u = torch.from_numpy(labels[train_idx_u]).float()
            dataset_u = torch.utils.data.TensorDataset(X_train_u, y_train_u)
            loader_u = DataLoader(dataset_u, batch_size=min(args.batch_size, len(train_idx_u)), shuffle=True)

            X_test_u = torch.from_numpy(contexts[test_idx_u]).float() if len(test_idx_u) else None
            y_test_u = torch.from_numpy(labels[test_idx_u]).float() if len(test_idx_u) else None

            model_u = MovielensMLP(in_dim=contexts.shape[1]).to(args.device)
            opt_u = optim.Adam(model_u.parameters(), lr=args.lr)
            # Star-shaped fine-tuning: initialize from first user's weights if available
            if base_sd is not None:
                model_u.load_state_dict(base_sd)
            else:
                import copy
                base_sd = copy.deepcopy(model_u.state_dict())  # save first model's parameters
            loss_fn = nn.MSELoss() if args.task == "regression" else nn.BCEWithLogitsLoss()

            for ep in range(args.epochs):
                running_loss = 0.0
                for xb, yb in loader_u:
                    xb = xb.to(args.device)
                    yb = yb.to(args.device)
                    outs = model_u(xb)
                    if args.task == "classification":
                        loss = loss_fn(outs, yb)
                    else:
                        loss = loss_fn(outs, yb)
                    opt_u.zero_grad()
                    loss.backward()
                    opt_u.step()
                    running_loss += loss.item() * xb.size(0)

                avg_loss = running_loss / len(dataset_u)
                with torch.no_grad():
                    preds_epoch = model_u(X_train_u.to(args.device))
                    if args.task == "classification":
                        metric = ((torch.sigmoid(preds_epoch) > 0.5).float().cpu() == y_train_u).float().mean().item()
                        metric_name = "train_acc"
                        metric_val = metric * 100.0
                    else:
                        metric = nn.functional.mse_loss(preds_epoch.cpu(), y_train_u).item()
                        metric_name = "train_mse"
                        metric_val = metric
                if args.task == "classification":
                    print(f"[User {uid}] epoch {ep+1}/{args.epochs} loss={avg_loss:.4f}  train_acc={metric_val:.2f}%")
                else:
                    print(f"[User {uid}] epoch {ep+1}/{args.epochs} loss={avg_loss:.4f}  train_mse={metric_val:.6f}")

                test_msg = ""
                if X_test_u is not None and len(test_idx_u):
                    with torch.no_grad():
                        preds_test = model_u(X_test_u.to(args.device))
                        if args.task == "classification":
                            acc_test = (torch.sigmoid(preds_test) > 0.5).float().eq(y_test_u.to(args.device)).float().mean().item()
                            test_msg = f"  test_acc={acc_test*100:.2f}%"
                        else:
                            mse_test = nn.functional.mse_loss(preds_test, y_test_u.to(args.device)).item()
                            test_msg = f"  test_mse={mse_test:.6f}"
                print(f"[User {uid}] trained{test_msg}")

            per_user_states.append({k: v.cpu() for k, v in model_u.state_dict().items()})

        torch.save(per_user_states, args.mlp_out)
        print(f"Saved per-user MLP state_dict list to {args.mlp_out} (n_users={len(per_user_states)})")
    else:
        print(f"{args.mlp_out} exists, skipping per-user MLP training.")

    if not args.linear_out.exists():
        lin_thetas = sample_linear_thetas(ctx_train, y_train, n_samples=args.n_linear, subset_size=args.subset_size)
        np.save(args.linear_out, lin_thetas)
        print(f"Saved linear theta samples to {args.linear_out} with shape {lin_thetas.shape}")
    else:
        print(f"{args.linear_out} exists, skipping linear sampling.")

    np.savez(
        args.dataset_out,
        contexts=contexts.astype(np.float32),
        labels=labels.astype(np.int64),
        entries=entries.astype(np.int64),
        user_features=user_features.astype(np.float32),
        movie_features=movie_features.astype(np.float32),
    )
    print(f"Cached contexts + embeddings at {args.dataset_out}")


if __name__ == "__main__":
    main()
