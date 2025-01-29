import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Union, Sequence, Dict
from modeling_llama3_SAE import TopK, Autoencoder, JumpReLu, HeavyStep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk


# ------------------------------------------------------
#               Support Loss Functions
# ------------------------------------------------------

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error per example (shape: [batch])
    """
    return ((reconstruction - original_input) ** 2).mean(dim=1) / (
        original_input**2
    ).mean(dim=1)


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss per example (shape: [batch])
    """
    return latent_activations.abs().sum(dim=1) / (original_input.norm(dim=1) + 1e-9)


def double_well_loss(latent_activation: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


# ------------------------------------------------------
#               Dataclass Configuration
# ------------------------------------------------------


@dataclass
class SAE_Train_config:
    name: str = field(init=False)
    activation: str = "topk"
    k: int = 32
    n_inputs: int = 2048
    factor: int = 3
    n_latents: int = field(init=False)
    lr: float = 1e-4
    max_epochs: int = 50
    batch_size: int = 2048
    ds_path: str = ""
    condition: bool = True
    sparstity_coef: float = 1.0
    sparstity_coef_warmup_duration: float = 0.0
    ckpt_path: str = None
    cond_loss_scaling: float = (
        1.0  # <--- Replacing context.get_hparam("cond_loss_scaling")
    )

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        self.n_latents = self.n_inputs * self.factor
        model = self.ds_path.split("/")[-1].split("-")[0]
        block = self.ds_path.split("/")[-1].split("-")[2][1:]
        if self.activation == "topk":
            self.name = f"{model}-l{self.n_latents}-b{block}-k{self.k}"
        else:
            self.name = f"{model}-l{self.n_latents}-b{block}-{self.activation}-s{self.sparstity_coef}"


# ------------------------------------------------------
#                The "Trial" Class
# ------------------------------------------------------
# Although we've removed Determined, let's keep the same logic
# by creating a class that holds model, optimizer, and the
# train/eval steps, etc.


class SAETrial:
    def __init__(self, conf: SAE_Train_config, device: str = "cuda"):
        """
        Initialize the SAETrial with the same logic from your original code,
        but no references to Determined contexts.
        """
        self.conf = conf
        self.device = device

        # Load the dataset
        logging.info(f"Loading dataset from: {self.conf.ds_path}")
        dataset = load_from_disk(self.conf.ds_path)
        self.dm = dataset.with_format("torch")

        # Build the activation
        if conf.activation == "topk":
            logging.info(f"Using activation = TopK (k={conf.k})")
            activation = TopK(conf.k)
        elif conf.activation == "topk-sigmoid":
            logging.info(f"Using activation = TopK + Sigmoid (k={conf.k})")
            activation = TopK(conf.k, nn.Sigmoid())
        elif conf.activation == "jumprelu":
            logging.info(f"Using activation = JumpReLu")
            activation = JumpReLu()
        elif conf.activation == "relu":
            logging.info(f"Using activation = ReLU")
            activation = nn.ReLU()
        else:
            raise NotImplementedError(
                f"Activation '{conf.activation}' not implemented."
            )

        AE = Autoencoder(
            n_inputs=conf.n_inputs,
            n_latents=conf.n_latents,
            activation=activation,
            tied=True,
            normalize=True,
        )

        # Optional checkpoint load
        if conf.ckpt_path is not None:
            logging.info(f"Loading checkpoint from {conf.ckpt_path}.")
            # Expecting a dict with "models_state_dict" -> [0]
            loaded = torch.load(conf.ckpt_path, map_location=device)
            sae_state_dict = loaded["models_state_dict"][0]
            AE.load_state_dict(sae_state_dict, strict=True)

        self.model = AE.to(device)

        # Set up the conditional vs. non-conditional losses:
        if conf.cond_loss_scaling > 0.0:
            if isinstance(self.model.activation, TopK):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_topk_c
            elif isinstance(self.model.activation, JumpReLu):
                self.heavy_step = HeavyStep()
                self.loss_fn = self.loss_jumprelu_c
            elif isinstance(self.model.activation, nn.ReLU):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_relu_c
        else:
            if isinstance(self.model.activation, TopK):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_topk
            elif isinstance(self.model.activation, JumpReLu):
                self.heavy_step = HeavyStep()
                self.loss_fn = self.loss_jumprelu
            elif isinstance(self.model.activation, nn.ReLU):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_relu

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr)

        # Setup for warmup steps
        total_train_steps = len(self.dm["train"]) * self.conf.max_epochs
        self.sparstity_coef = (
            self.conf.sparstity_coef if self.conf.sparstity_coef else 0.0
        )
        self.max_sparstity_coef = self.conf.sparstity_coef
        self.sparstity_coef_step = (
            self.conf.sparstity_coef
            / (total_train_steps * self.conf.sparstity_coef_warmup_duration + 1e-9)
            if self.conf.sparstity_coef_warmup_duration > 0
            else 0
        )
        self.step = 0
        self.sparstity_coef_warmup = int(
            total_train_steps * self.conf.sparstity_coef_warmup_duration
        )

    def get_sparsity_coef(self):
        """
        Ramp up the self.sparstity_coef if needed.
        """
        if self.step < self.sparstity_coef_warmup:
            self.sparstity_coef += self.sparstity_coef_step
        else:
            self.sparstity_coef = self.max_sparstity_coef

    def dummy(self, **kwargs):
        return torch.tensor(0.0, device=self.device)

    def cond_loss_fn(self, latents, toxicity):
        return F.binary_cross_entropy_with_logits(
            latents[:, 0], toxicity, reduction="none"
        )

    # --------------------
    #   Loss variants
    # --------------------
    def loss_topk_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = normalized_mean_squared_error(recons, acts)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return l2_loss + cond_loss, l1_loss, l2_loss, cond_loss, 0.0

    def loss_topk(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = normalized_mean_squared_error(recons, acts)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return l2_loss, l1_loss, l2_loss, cond_loss, 0.0

    def loss_jumprelu_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        heavy_step_loss = self.heavy_step(
            latents, torch.exp(self.model.threshold)
        ).mean(dim=1)

        return (
            l2_loss + cond_loss + self.sparstity_coef * heavy_step_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            heavy_step_loss,
        )

    def loss_jumprelu(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        heavy_step_loss = self.heavy_step(
            latents, torch.exp(self.model.threshold)
        ).mean(dim=1)

        return (
            l2_loss + self.sparstity_coef * heavy_step_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            heavy_step_loss,
        )

    def loss_relu_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return (
            l2_loss + cond_loss + self.sparstity_coef * l1_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            0.0,
        )

    def loss_relu(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return (
            l2_loss + self.sparstity_coef * l1_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            0.0,
        )

    def calc_sep_loss(self, loss, mask):
        """
        Separates the mean losses for the "toxic" portion and "non-toxic" portion
        (wherever mask is True vs. False).
        """
        e = 1e-9
        loss_mod_tox = loss * mask
        loss_tox = loss_mod_tox.sum() / ((loss_mod_tox > 0).sum() + e)
        loss_mod_ntox = loss * (mask != True)
        loss_ntox = loss_mod_ntox.sum() / ((loss_mod_ntox > 0).sum() + e)
        return loss_tox, loss_ntox

    def clip_grads(self, params):
        torch.nn.utils.clip_grad_norm_(params, 1.0)

    def grad_scaling(self, grad, tox):
        # scale gradient for toxic samples
        grad[tox > 0] *= self.conf.cond_loss_scaling
        return grad

    def train_batch(self, batch: TorchData):
        """
        Exactly replicates what used to be in `train_batch` under Determined,
        but as a plain function returning a dict of losses.
        """
        acts = batch["acts"].to(self.device)
        toxicity = batch["toxicity"].float().to(self.device)

        toxicity_mask = toxicity >= 0
        toxicity = toxicity * toxicity_mask

        latents_pre_act, latents, recons = self.model(acts)

        # Get the relevant loss function outputs
        loss, l1_loss, l2_loss, cond_loss_pre_rmse, heavy_step_loss = self.loss_fn(
            latents_pre_act, latents, recons, acts, toxicity
        )

        # Register a hook on the cond_loss_pre_rmse so we can scale gradient for toxic entries
        handle = cond_loss_pre_rmse.register_hook(
            lambda grad: self.grad_scaling(grad, toxicity)
        )

        # Take the mean across the batch for the main training objective
        loss_mean = loss.mean()
        l1_loss_mean = l1_loss.mean()
        l2_loss_mean = l2_loss.mean()

        # Calculate split losses
        loss_mean_t, loss_mean_nt = self.calc_sep_loss(loss, toxicity)
        l1_loss_mean_t, l1_loss_mean_nt = self.calc_sep_loss(l1_loss, toxicity)
        l2_loss_mean_t, l2_loss_mean_nt = self.calc_sep_loss(l2_loss, toxicity)
        cond_loss_mean_t, cond_loss_mean_nt = self.calc_sep_loss(
            cond_loss_pre_rmse, toxicity
        )

        cond_loss_pre_rmse_mean = cond_loss_pre_rmse.mean()
        heavy_step_loss_mean = (
            heavy_step_loss
            if isinstance(heavy_step_loss, float)
            else heavy_step_loss.mean()
        )

        # Backward & step
        self.optimizer.zero_grad()
        loss_mean.backward()
        max_grad = max(
            [
                p.grad.detach().abs().max()
                for p in self.model.parameters()
                if p.grad is not None
            ]
        )
        min_grad = min(
            [
                p.grad.detach().abs().min()
                for p in self.model.parameters()
                if p.grad is not None
            ]
        )

        self.clip_grads(self.model.parameters())
        self.optimizer.step()

        # remove the hook
        handle.remove()

        return {
            "loss": loss_mean.item(),
            "loss_t": loss_mean_t.item(),
            "loss_nt": loss_mean_nt.item(),
            "l1_loss": l1_loss_mean.item(),
            "l1_loss_t": l1_loss_mean_t.item(),
            "l1_loss_nt": l1_loss_mean_nt.item(),
            "l2_loss": l2_loss_mean.item(),
            "l2_loss_t": l2_loss_mean_t.item(),
            "l2_loss_nt": l2_loss_mean_nt.item(),
            "heavy step": heavy_step_loss_mean.item()
            if not isinstance(heavy_step_loss_mean, float)
            else heavy_step_loss_mean,
            "cond_loss": cond_loss_pre_rmse_mean.item(),
            "cond_loss_t": cond_loss_mean_t.item(),
            "cond_loss_nt": cond_loss_mean_nt.item(),
            "max_grad": max_grad.item(),
            "min_grad": min_grad.item(),
        }

    def evaluate_batch(self, batch: TorchData):
        """
        Exactly replicates what used to be in `evaluate_batch` under Determined,
        but as a plain function returning a dict of evaluation losses.
        """
        acts = batch["acts"].to(self.device)
        toxicity = batch["toxicity"].float().to(self.device)

        toxicity_mask = toxicity >= 0
        toxicity = toxicity * toxicity_mask

        with torch.no_grad():
            latents_pre_act, latents, recons = self.model(acts)
            loss, l1_loss, l2_loss, cond_loss_pre_rmse, heavy_step_loss = self.loss_fn(
                latents_pre_act, latents, recons, acts, toxicity
            )

        loss_mean = loss.mean()
        l1_loss_mean = l1_loss.mean()
        l2_loss_mean = l2_loss.mean()

        loss_mean_t, loss_mean_nt = self.calc_sep_loss(loss, toxicity)
        l1_loss_mean_t, l1_loss_mean_nt = self.calc_sep_loss(l1_loss, toxicity)
        l2_loss_mean_t, l2_loss_mean_nt = self.calc_sep_loss(l2_loss, toxicity)
        cond_loss_mean_t, cond_loss_mean_nt = self.calc_sep_loss(
            cond_loss_pre_rmse, toxicity
        )
        cond_loss_pre_rmse_mean = cond_loss_pre_rmse.mean()
        heavy_step_loss_mean = (
            heavy_step_loss
            if isinstance(heavy_step_loss, float)
            else heavy_step_loss.mean()
        )

        return {
            "val_loss": loss_mean.item(),
            "val_l1_loss": l1_loss_mean.item(),
            "val_l2_loss": l2_loss_mean.item(),
            "val_heavy step": heavy_step_loss_mean.item()
            if not isinstance(heavy_step_loss_mean, float)
            else heavy_step_loss_mean,
            "val_cond_loss": cond_loss_pre_rmse_mean.item(),
            "val_loss_t": loss_mean_t.item(),
            "val_l1_loss_t": l1_loss_mean_t.item(),
            "val_l2_loss_t": l2_loss_mean_t.item(),
            "val_cond_loss_t": cond_loss_mean_t.item(),
            "val_loss_nt": loss_mean_nt.item(),
            "val_l1_loss_nt": l1_loss_mean_nt.item(),
            "val_l2_loss_nt": l2_loss_mean_nt.item(),
            "val_cond_loss_nt": cond_loss_mean_nt.item(),
        }

    def build_training_data_loader(self):
        """
        Replacement for Determined's build_training_data_loader.
        """
        return DataLoader(
            self.dm["train"],
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def build_validation_data_loader(self):
        """
        Replacement for Determined's build_validation_data_loader.
        """
        return DataLoader(
            self.dm["test"],
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=4,
        )


# ------------------------------------------------------
#               Main Training Loop
# ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run SAE training without Determined.")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g. 'cuda' or 'cpu').",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Load config
    with open(args.config_path, "r") as f:
        conf_as_json = json.load(f)
    conf = SAE_Train_config(**conf_as_json)

    # Build the trial
    trial = SAETrial(conf, device=args.device)

    # DataLoaders
    train_loader = trial.build_training_data_loader()
    val_loader = trial.build_validation_data_loader()

    logging.info(f"Starting training for {conf.max_epochs} epochs.")

    for epoch_idx in range(conf.max_epochs):
        # --- Training ---
        trial.model.train()
        for batch_idx, batch in enumerate(train_loader):
            trial.step += 1
            # Warmup for sparsity if needed
            trial.get_sparsity_coef()

            metrics = trial.train_batch(batch)
            if (batch_idx + 1) % 50 == 0:
                logging.info(
                    f"Epoch {epoch_idx + 1}, Batch {batch_idx + 1}, "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"L1: {metrics['l1_loss']:.4f}, L2: {metrics['l2_loss']:.4f}"
                )

        # --- Validation ---
        trial.model.eval()
        val_metrics_list = []
        for batch in val_loader:
            vm = trial.evaluate_batch(batch)
            val_metrics_list.append(vm)

        # Compute mean of val metrics
        if len(val_metrics_list) > 0:
            mean_vals = {}
            for k in val_metrics_list[0].keys():
                mean_vals[k] = sum(d[k] for d in val_metrics_list) / len(
                    val_metrics_list
                )
            logging.info(
                f"[Epoch {epoch_idx + 1}/{conf.max_epochs}] "
                f"Validation Loss: {mean_vals['val_loss']:.4f} | "
                f"Val L1: {mean_vals['val_l1_loss']:.4f}, "
                f"Val L2: {mean_vals['val_l2_loss']:.4f}"
            )

        ## SAVE CHECKPOINT
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": trial.model.state_dict(),
                "optimizer_state_dict": trial.optimizer.state_dict(),
            },
            f"checkpoint_epoch_{epoch_idx + 1}.pt",
        )


if __name__ == "__main__":
    main()
