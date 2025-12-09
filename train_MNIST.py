import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # added for MLP training
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MNISTMLP(nn.Module):
    """Simple MLP classifier whose penultimate layer has dimension `embed_dim`."""

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_net = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.embed_net(x)
        logits = self.classifier(h)
        return logits, h


@torch.no_grad()
def extract_features(model: MNISTMLP, loader: DataLoader, device: torch.device):
    model.eval()
    features_list, labels_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        _, feats = model(imgs)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def train_classifier(epochs: int = 5, batch_size: int = 256, lr: float = 1e-3, device: str = 'cpu', save_path: str = 'Results_mnist/best_mnist_mlp.pth'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MNISTMLP(embed_dim=8).to(device)

    # If pretrained weights exist, load and skip training
    if Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded pretrained model from {save_path}. Skipping training.")
        # Create minimal dummy loaders to keep return signature
        empty_loader = DataLoader([], batch_size=1)
        return model, empty_loader, empty_loader

    best_val_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: loss={epoch_loss/total:.4f}  train_acc={train_acc:.2f}%")

        # -------- Evaluation on the test set to monitor overfitting --------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs_val, labels_val in test_loader:
                imgs_val, labels_val = imgs_val.to(device), labels_val.to(device)
                logits_val, _ = model(imgs_val)
                preds_val = logits_val.argmax(dim=1)
                val_correct += (preds_val == labels_val).sum().item()
                val_total += imgs_val.size(0)
        val_acc = 100.0 * val_correct / val_total
        print(f"                val_acc={val_acc:.2f}%")

        # -------- Save best model --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"                Saved new best model to {save_path}")
        model.train()

    # -------- Load best model for final evaluation --------
    model.load_state_dict(torch.load(save_path))
    print(f"Loaded best model (val_acc={best_val_acc:.2f}%) for final test evaluation")

    # Quick evaluation on test set
    model.eval()
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    print(f"Test accuracy: {100.0*correct/total:.2f}%")
    return model, train_loader, test_loader


def generate_theta_distribution(features: np.ndarray, labels: np.ndarray, n_samples: int = 10000, d: int = 8, subset_size: int = 20):
    assert features.shape[1] == d, "Feature dimension mismatch"
    theta_samples = np.zeros((n_samples, d), dtype=np.float32)
    mse_list = np.zeros(n_samples, dtype=np.float32)
    acc_list = np.zeros(n_samples, dtype=np.float32)
    pos_labels = np.zeros(n_samples, dtype=np.int32)
    label_indices = {lbl: np.where(labels == lbl)[0] for lbl in range(10)}
    others_indices = {lbl: np.where(labels != lbl)[0] for lbl in range(10)}

    for i in tqdm(range(n_samples), desc="Sampling θ"):
        pos = np.random.randint(10)
        pos_labels[i] = pos
        pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
        neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        X = features[idx]  # (subset_size, d)
        y = np.concatenate([np.ones(subset_size // 2), -np.ones(subset_size // 2)])
        # Fit linear model via least squares with small ridge for stability
        XtX = X.T @ X + 1e-6 * np.eye(d)
        theta = np.linalg.solve(XtX, X.T @ y)
        theta_samples[i] = theta

        # ---------- metrics ----------
        y_pred = X @ theta  # shape (subset_size,)
        mse_list[i] = np.mean((y_pred - y) ** 2)
        acc_list[i] = np.mean(np.sign(y_pred) == y)
    # Print summary statistics for quick inspection (optional)
    print(f"θ sampling finished. Avg MSE={mse_list.mean():.4f}, Avg acc={acc_list.mean()*100:.2f}%")

    # ---- t-SNE visualization ----
    tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    theta_2d = tsne.fit_transform(theta_samples)
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(theta_2d[:,0], theta_2d[:,1], c=pos_labels, cmap='tab10', s=4, alpha=0.8)
    plt.colorbar(scatter, ticks=range(10), label='Positive digit')
    plt.title('t-SNE of θ samples colored by positive class')
    plt.tight_layout()
    plt.savefig('mnist_theta_tsne.pdf')
    plt.close()

    return theta_samples, mse_list, acc_list, pos_labels

# =====================  NEW FUNCTIONS for raw-image binary models =====================

def load_full_mnist(flatten: bool = True):
    """Load the full (train+test) MNIST dataset as tensors.

    Returns
    -------
    images : torch.FloatTensor  (N, 784) or (N,1,28,28)
    labels : torch.LongTensor   (N,)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    full_ds = torch.utils.data.ConcatDataset([train_ds, test_ds])
    loader = DataLoader(full_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    imgs_list, lbls_list = [], []
    for imgs, lbls in loader:
        if flatten:
            imgs = imgs.view(imgs.size(0), -1)
        imgs_list.append(imgs)
        lbls_list.append(lbls)
    images = torch.cat(imgs_list, dim=0)
    labels = torch.cat(lbls_list, dim=0)
    return images, labels


def train_linear_models(images: torch.Tensor, labels: torch.Tensor, *, n_models: int = 10000, subset_size: int = 20, ridge: float = 1e-6):
    """Sample subsets and fit linear models via least squares.

    Each model is trained on `subset_size` examples where half are of one randomly
    chosen positive digit (labelled +1) and the rest are other digits (labelled -1).

    Parameters
    ----------
    images : torch.Tensor (N, D)
    labels : torch.Tensor (N,)
    n_models : int
    subset_size : int
    ridge : float

    Returns
    -------
    thetas : np.ndarray (n_models, D)
    pos_digits : np.ndarray (n_models,)
    """
    N, D = images.shape
    imgs_np = images.numpy()
    lbls_np = labels.numpy()

    label_indices = {lbl: np.where(lbls_np == lbl)[0] for lbl in range(10)}
    others_indices = {lbl: np.where(lbls_np != lbl)[0] for lbl in range(10)}

    thetas = np.zeros((n_models, D), dtype=np.float32)
    pos_digits = np.zeros(n_models, dtype=np.int32)
    accs = np.zeros(n_models, dtype=np.float32)
    mses = np.zeros(n_models, dtype=np.float32)

    for i in tqdm(range(n_models), desc="Linear models"):
        pos = np.random.randint(10)
        pos_digits[i] = pos
        pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
        neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        X = imgs_np[idx]  # (subset_size, D)
        y = np.concatenate([np.ones(subset_size // 2), -np.ones(subset_size // 2)])
        XtX = X.T @ X + ridge * np.eye(D)
        theta = np.linalg.solve(XtX, X.T @ y)
        thetas[i] = theta.astype(np.float32)

        # ----- compute train metrics -----
        y_pred = X @ theta
        mse_train = np.mean((y_pred - y) ** 2)
        acc_train = np.mean(np.sign(y_pred) == y)

        # ----- test subset -----
        test_pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
        test_neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
        test_idx = np.concatenate([test_pos_idx, test_neg_idx])
        X_test = imgs_np[test_idx]
        y_test = np.concatenate([np.ones(subset_size // 2), -np.ones(subset_size // 2)])
        y_test_pred = X_test @ theta
        acc_test = np.mean(np.sign(y_test_pred) == y_test)

        mses[i] = mse_train
        accs[i] = acc_train
        print(f"[Linear {i+1}/{n_models}] pos_digit={pos}  train_acc={acc_train*100:.2f}%  test_acc={acc_test*100:.2f}%  mse={mse_train:.4f}")

    return thetas, pos_digits, accs, mses


class SmallMLP(nn.Module):
    """Simple binary MLP: 784 -> 128 -> 64 -> 1"""

    def __init__(self, in_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # output shape (batch,)


def train_mlp_models(images: torch.Tensor, labels: torch.Tensor, *, n_models: int = 10000, subset_size: int = 20, device: str = 'cpu', epochs: int = 20, lr: float = 1e-2):
    """Train n_models small MLPs on random ±1 tasks as above.

    Returns list of state_dicts and pos_digits array.
    """
    # Keep data on CPU; move mini-batches to GPU as needed
    N, D = images.shape
    state_dicts = []
    base_sd = None  # weights of the FIRST trained model

    # Pre-compute indices per label on CPU numpy for sampling speed
    lbls_np = labels.cpu().numpy()
    label_indices = {lbl: np.where(lbls_np == lbl)[0] for lbl in range(10)}
    others_indices = {lbl: np.where(lbls_np != lbl)[0] for lbl in range(10)}

    for i in tqdm(range(n_models), desc="MLP models"):
        pos = np.random.randint(10)
        pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
        neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])

        X_cpu = images[idx]  # still on CPU
        y_cpu = torch.cat([torch.ones(subset_size // 2), -torch.ones(subset_size // 2)])

        dataset = torch.utils.data.TensorDataset(X_cpu, y_cpu)
        loader = DataLoader(dataset, batch_size=min(32, subset_size), shuffle=True)

        model = SmallMLP(in_dim=D).to(device)
        if base_sd is not None:
            model.load_state_dict(base_sd)  # star-shaped fine-tuning: always start from first model
        optim_m = optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = F.mse_loss(preds, yb)
                optim_m.zero_grad()
                loss.backward()
                optim_m.step()

        with torch.no_grad():
            preds_final = model(X_cpu.to(device))
            acc_train = (torch.sign(preds_final) == y_cpu.to(device)).float().mean().item()

            # ----- test subset -----
            test_pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
            test_neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
            test_idx = np.concatenate([test_pos_idx, test_neg_idx])
            X_test_cpu = images[test_idx]
            y_test_cpu = torch.cat([torch.ones(subset_size // 2), -torch.ones(subset_size // 2)])
            preds_test = model(X_test_cpu.to(device))
            acc_test = (torch.sign(preds_test) == y_test_cpu.to(device)).float().mean().item()

        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        state_dicts.append(sd_cpu)

        # Save first model's weights as base for all subsequent models
        if base_sd is None:
            from copy import deepcopy
            base_sd = deepcopy(sd_cpu)

        print(f"[MLP   {i+1}/{n_models}] pos_digit={pos}  train_acc={acc_train*100:.2f}%  test_acc={acc_test*100:.2f}%  loss={loss.item():.4f}")

    return state_dicts


def train_mlp_models_list(images: torch.Tensor, labels: torch.Tensor, *, n_models: int = 10000, subset_size: int = 20, device: str = 'cpu', epochs: int = 20, lr: float = 1e-2):
    """Train n_models small MLPs on random ±1 tasks as above, each starting from a random initialization.

    Returns list of state_dicts.
    """
    # Keep data on CPU; move mini-batches to GPU as needed
    N, D = images.shape
    state_dicts = []

    # Pre-compute indices per label on CPU numpy for sampling speed
    lbls_np = labels.cpu().numpy()
    label_indices = {lbl: np.where(lbls_np == lbl)[0] for lbl in range(10)}
    others_indices = {lbl: np.where(lbls_np != lbl)[0] for lbl in range(10)}

    for i in tqdm(range(n_models), desc="MLP models (random init)"):
        pos = np.random.randint(10)
        pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
        neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])

        X_cpu = images[idx]  # still on CPU
        y_cpu = torch.cat([torch.ones(subset_size // 2), -torch.ones(subset_size // 2)])

        dataset = torch.utils.data.TensorDataset(X_cpu, y_cpu)
        loader = DataLoader(dataset, batch_size=min(32, subset_size), shuffle=True)

        # Each model starts from a fresh random initialization
        model = SmallMLP(in_dim=D).to(device)
        optim_m = optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = F.mse_loss(preds, yb)
                optim_m.zero_grad()
                loss.backward()
                optim_m.step()

        with torch.no_grad():
            preds_final = model(X_cpu.to(device))
            acc_train = (torch.sign(preds_final) == y_cpu.to(device)).float().mean().item()

            # ----- test subset -----
            test_pos_idx = np.random.choice(label_indices[pos], subset_size // 2, replace=False)
            test_neg_idx = np.random.choice(others_indices[pos], subset_size // 2, replace=False)
            test_idx = np.concatenate([test_pos_idx, test_neg_idx])
            X_test_cpu = images[test_idx]
            y_test_cpu = torch.cat([torch.ones(subset_size // 2), -torch.ones(subset_size // 2)])
            preds_test = model(X_test_cpu.to(device))
            acc_test = (torch.sign(preds_test) == y_test_cpu.to(device)).float().mean().item()

        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        state_dicts.append(sd_cpu)

        print(f"[MLP-R {i+1}/{n_models}] pos_digit={pos}  train_acc={acc_train*100:.2f}%  test_acc={acc_test*100:.2f}%  loss={loss.item():.4f}")

    return state_dicts


def main():
    parser = argparse.ArgumentParser(description="Train MNIST MLP and sample θ distribution.")
    parser.add_argument('--n_models', type=int, default=10000, help='Number of linear/MLP models to train')
    parser.add_argument('--subset_size', type=int, default=100, help='Subset size for each model (half positive, half negative)')
    parser.add_argument('--linear_out', type=str, default='Results/mnist_linear_thetas.npy', help='Output .npy for linear model parameters')
    parser.add_argument('--mlp_out', type=str, default='Results/mnist_mlp_state_dicts.pt', help='Output file for list of MLP state_dicts')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(0)

    # ------------- Step 1: Load full MNIST images+labels -------------
    images, labels = load_full_mnist(flatten=True)

    # ------------- Step 2: Train (or load) linear models -------------
    lin_path = Path(args.linear_out)
    lin_path.parent.mkdir(parents=True, exist_ok=True)

    if lin_path.exists():
        print(f"{lin_path} already exists, skipping linear model training.")
        thetas = np.load(lin_path)
    else:
        thetas, pos_digits_linear, _, __ = train_linear_models(
            images, labels, n_models=args.n_models, subset_size=args.subset_size)
        np.save(lin_path, thetas)
        print(f"Saved linear model parameters to {lin_path} with shape {thetas.shape}.")

    # ------------- Step 3: Train (or load) MLP models ----------------
    mlp_path = Path(args.mlp_out)
    mlp_path.parent.mkdir(parents=True, exist_ok=True)

    if mlp_path.exists():
        print(f"{mlp_path} already exists, skipping MLP training.")
        state_dicts = torch.load(mlp_path)
    else:
        state_dicts = train_mlp_models(images, labels, n_models=args.n_models, subset_size=args.subset_size, device=args.device)
        torch.save(state_dicts, mlp_path)
        print(f"Saved MLP model state_dicts to {mlp_path} (list length={len(state_dicts)}).")


if __name__ == '__main__':
    main()
