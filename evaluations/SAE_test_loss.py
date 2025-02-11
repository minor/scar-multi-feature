import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

# -------------------------------------------------------------------------
# 1. Bring over the same pieces from your training code
# -------------------------------------------------------------------------


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: [batch, n_inputs]
    :param original_input: [batch, n_inputs]
    :return: normalized MSE per example, shape [batch]
    """
    return ((reconstruction - original_input) ** 2).mean(dim=1) / (
        (original_input**2).mean(dim=1) + 1e-9
    )


class TopK(nn.Module):
    """
    implement TopK
    """

    def __init__(self, k, post_act=None):
        super().__init__()
        self.k = k
        self.post_act = post_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_latents]
        # zero out everything except top-k
        # (per-row top-k).
        with torch.no_grad():
            # topk returns (values, indices). We'll gather mask:
            val, idx = torch.topk(x, self.k, dim=1)
            mask = torch.zeros_like(x).scatter_(1, idx, 1.0)
        out = x * mask
        if self.post_act is not None:
            out = self.post_act(out)
        return out


class JumpReLu(nn.Module):
    """Kept for completeness if needed."""

    def forward(self, x: torch.Tensor):
        return torch.relu(x)


class Autoencoder(nn.Module):
    """
    Matches the `Autoencoder` in your training code.
    - 'tied=True' means the decoder is transposed version of encoder or similar
      if you implemented it that way.
    - 'normalize=True' means you might do something to normalize the input, etc.

    Adjust dimensions or forward logic if your code differs.
    """

    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        activation: nn.Module,
        tied: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.activation = activation
        self.normalize = normalize

        # Example linear encoder/decoder
        # If "tied=True," you might do something like sharing weights,
        # but here is a simple version:
        self.encoder = nn.Linear(n_inputs, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, n_inputs, bias=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: [batch, n_inputs]
        :return: (latents_pre_act, latents, recons)
        """
        # latents_pre_act: pre-activation
        latents_pre_act = self.encoder(x)
        # latents: apply activation
        latents = self.activation(latents_pre_act)
        # recons: decode
        recons = self.decoder(latents)
        return latents_pre_act, latents, recons


# -------------------------------------------------------------------------
# 2. Main Testing Script
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test reconstruction loss of SAE.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. checkpoint_epoch_100.pt).",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="Path to the test dataset (saved via datasets.save_to_disk).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--plot_output", type=str, default="reconstruction_loss_hist.png"
    )
    # Model architecture args (must match training)
    parser.add_argument(
        "--k", type=int, default=32, help="TopK value if using topk activation."
    )
    parser.add_argument(
        "--n_inputs", type=int, default=2048, help="Dimension of input (acts)."
    )
    parser.add_argument(
        "--factor", type=int, default=3, help="Factor => n_latents = n_inputs * factor."
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="topk",
        choices=["topk", "jumprelu", "relu"],
        help="Activation type, same as training.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the dataset
    logging.info(f"Loading test dataset from: {args.test_dataset_path}")
    ds = load_from_disk(args.test_dataset_path)

    # If the dataset is a DatasetDict with ["test"], pick that. Otherwise assume a single dataset
    if isinstance(ds, dict) and "test" in ds:
        ds = ds["test"]

    ds.set_format("torch", columns=["acts"])
    test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 2. Build the same activation module
    if args.activation == "topk":
        logging.info(f"Using TopK activation with k={args.k}")
        activation_module = TopK(args.k)
    elif args.activation == "jumprelu":
        logging.info("Using JumpReLu activation.")
        activation_module = JumpReLu()
    elif args.activation == "relu":
        logging.info("Using standard ReLU activation.")
        activation_module = nn.ReLU()
    else:
        raise NotImplementedError(f"Unknown activation {args.activation}")

    n_latents = args.n_inputs * args.factor

    # 3. Instantiate the autoencoder
    logging.info(
        f"Instantiating Autoencoder with n_inputs={args.n_inputs}, n_latents={n_latents}."
    )
    model = Autoencoder(
        n_inputs=args.n_inputs,
        n_latents=n_latents,
        activation=activation_module,
        tied=True,
        normalize=True,
    )
    model.to(device)

    # 4. Load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    # Some training code uses "model_state_dict", others "models_state_dict"[0].
    # Adjust as needed:
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    elif "models_state_dict" in checkpoint_data and isinstance(
        checkpoint_data["models_state_dict"], list
    ):
        model.load_state_dict(checkpoint_data["models_state_dict"][0], strict=True)
    else:
        # fallback, try entire checkpoint_data if it's a state_dict
        model.load_state_dict(checkpoint_data, strict=True)

    model.eval()

    # 5. Evaluate reconstruction losses on test set
    all_losses = []

    with torch.no_grad():
        for batch in test_loader:
            acts = batch["acts"].to(device)  # shape [batch, n_inputs]
            # forward
            _, _, recons = model(acts)
            # compute per-sample normalized MSE
            mse_vec = normalized_mean_squared_error(recons, acts)  # shape [batch]
            all_losses.extend(mse_vec.cpu().numpy().tolist())

    # Compute average
    all_losses = np.array(all_losses)
    avg_loss = all_losses.mean()
    logging.info(f"Average reconstruction loss over test set: {avg_loss:.6f}")

    # 6. Optionally plot histogram of the reconstruction loss distribution
    plt.figure(figsize=(7, 5))
    plt.hist(all_losses, bins=50, alpha=0.7, edgecolor="black")
    plt.title("Reconstruction Loss Distribution (Normalized MSE)")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.axvline(avg_loss, color="red", linestyle="--", label=f"Mean={avg_loss:.5f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_output)
    logging.info(f"Saved histogram to {args.plot_output}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
