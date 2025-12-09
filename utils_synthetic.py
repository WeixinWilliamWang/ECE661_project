import numpy as np
import matplotlib.pyplot as plt
import os, warnings

def generate_s0_dataset(problem: str, n_pretrain: int, d: int = 2) -> np.ndarray:
    """Replicates the synthetic θ sampling logic (from original Synthetic.py).

    Parameters
    ----------
    problem : str
        One of {cross,rays,triangles,swirl,H,corners}
    n_pretrain : int
        Number of θ samples to generate.
    d : int, default 2
        Dimensionality (fixed at 2 in original experiments).
    """
    rng = np.random.default_rng()
    S0 = np.zeros((n_pretrain, d))

    if problem == "cross":
        num_mix = 2
        p0 = np.ones(num_mix) / num_mix
        theta0 = np.zeros((num_mix, d))
        Sigma0 = np.zeros((num_mix, d, d))
        Sigma0[0] = np.asarray([[1, 0.99], [0.99, 1]])
        Sigma0[1] = np.asarray([[1, -0.9], [-0.9, 1]])
        for i in range(n_pretrain):
            comp = rng.choice(num_mix, p=p0)
            S0[i] = rng.multivariate_normal(theta0[comp], Sigma0[comp])

    elif problem == "rays":
        for i in range(n_pretrain):
            while True:
                s0 = 6 * (rng.random(d) - 0.5)
                s0_norm = s0 / np.linalg.norm(s0)
                accept = (np.linalg.norm(s0) < 3) and (
                    (s0_norm @ np.array([1, 0]) > 0.95) or
                    (s0_norm @ np.array([-1, 0]) > 0.95) or
                    (s0_norm @ np.array([0, 1]) > 0.95) or
                    (s0_norm @ np.array([0, -1]) > 0.95)
                )
                if accept:
                    S0[i] = s0; break

    elif problem == "triangles":
        for i in range(n_pretrain):
            while True:
                s0 = 6 * (rng.random(d) - 0.5)
                accept = ((s0[0] > 0) and (abs(s0[1]) < abs(s0[0]))) or \
                         ((s0[0] < 0) and (abs(s0[1]) < abs(s0[0]) / 3))
                if accept:
                    S0[i] = s0; break

    elif problem == "swirl":
        for i in range(n_pretrain):
            u = rng.random()
            angle = 2 * np.pi * u
            radius = 3.5 * u
            width = u
            s0 = np.array([np.sin(angle), np.cos(angle)]) * (radius - width * (rng.random() - 0.5))
            s0 += np.array([1, -1])
            S0[i] = s0

    elif problem == "H":
        for i in range(n_pretrain):
            while True:
                s0 = 6 * (rng.random(d) - 0.5)
                if (s0[0] < -2.5) or (s0[0] > 2.5) or ((s0[1] > -0.25) and (s0[1] < 0.25)):
                    S0[i] = s0; break

    elif problem == "corners":
        for i in range(n_pretrain):
            while True:
                s0 = 6 * (rng.random(d) - 0.5)
                if ((abs(s0[0]) > 2.5) and (abs(s0[1]) > 1)) or ((abs(s0[1]) > 2.5) and (abs(s0[0]) > 1)):
                    S0[i] = s0; break
    else:
        raise ValueError(f"Unknown problem '{problem}'")

    return S0

######################## Visualization helpers ########################

def visualize_prior_scatter(real: np.ndarray, prior: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(prior[:, 0], prior[:, 1], s=4, c="lightgrey", alpha=0.6, label="Diffusion Prior")
    plt.scatter(real[:, 0], real[:, 1], s=8, c="blue", alpha=0.8, label="True")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(path); plt.close()


def visualize_prior_umap(real: np.ndarray, prior: np.ndarray, path: str):
    try:
        import umap.umap_ as umap
    except ImportError:
        warnings.warn("umap-learn not installed; skip UMAP viz")
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0)
    real_2d = reducer.fit_transform(real)
    prior_2d = reducer.transform(prior)
    plt.figure(figsize=(6, 6))
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], s=4, c="lightgrey", alpha=0.6, label="Diffusion Prior")
    plt.scatter(real_2d[:, 0], real_2d[:, 1], s=8, c="blue", alpha=0.8, label="True Prior")
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(path); plt.close()



def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (0.5, 2.5)
  else:
    return (None, None)
