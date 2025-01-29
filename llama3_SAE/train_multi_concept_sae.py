import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Union, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from modeling_llama3_SAE import TopK, Autoencoder, JumpReLu, HeavyStep

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


# ------------------------------------------------------
#         Dataclass Configuration w/ multi-concept
# ------------------------------------------------------


@dataclass
class SAE_Train_config:
    """
    This dataclass includes both the original fields and the new fields
    for multi-concept usage.
    """

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
    cond_loss_scaling: float = 1.0

    # New parameters for multi-concept
    num_concepts: int = 2  # Number of conditioned features
    concept_names: tuple = ("toxicity", "shakespeare")
    concept_loss_weights: tuple = (1.0, 1.0)  # Weight for each concept loss

    def __post_init__(self):
        self.n_latents = self.n_inputs * self.factor
        model = self.ds_path.split("/")[-1].split("-")[0]
        block = self.ds_path.split("/")[-1].split("-")[2][1:]
        self.name = f"{model}-l{self.n_latents}-b{block}-mc{self.num_concepts}"

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------
#               The "Trial" Class
# ------------------------------------------------------


class SAETrial:
    def __init__(self, conf: SAE_Train_config, device: str = "cuda"):
        """
        Initialize the SAETrial with multi-concept features, while
        preserving relevant comments and logic from the original code.
        """
        self.conf = conf
        self.device = device

        # Load the dataset (multi-concept dataset)
        logging.info(f"Loading dataset from: {self.conf.ds_path}")
        dataset = load_from_disk(self.conf.ds_path)
        self.dm = dataset.with_format("torch")

        # Build the activation
        if conf.activation == "topk":
            logging.info(f"Using activation = TopK (k={conf.k})")
            activation = TopK(conf.k)
        elif conf.activation == "topk-sigmoid":
            # Kept for completeness
            logging.info(f"Using activation = TopK + Sigmoid (k={conf.k})")
            activation = TopK(conf.k, nn.Sigmoid())
        elif conf.activation == "jumprelu":
            logging.info("Using activation = JumpReLu")
            activation = JumpReLu()
        elif conf.activation == "relu":
            logging.info("Using activation = ReLU")
            activation = nn.ReLU()
        else:
            raise NotImplementedError(
                f"Activation '{conf.activation}' not implemented."
            )

        # Initialize the Autoencoder
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

        # Multi-concept: define masks for each concept
        self.concept_masks = {
            "toxicity": (lambda b: b["toxicity"] >= 0),  # Example usage
            "shakespeare": (lambda b: b["label"] >= 0),  # Example usage
        }

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr)

        # Loss scaling setup
        total_steps = len(self.dm["train"]) * conf.max_epochs
        self.sparsity_coef_step = (
            conf.sparstity_coef / (total_steps * conf.sparstity_coef_warmup_duration)
            if conf.sparstity_coef_warmup_duration > 0
            else 0
        )
        self.step = 0

        # # Setup for warmup steps
        # total_train_steps = len(self.dm["train"]) * self.conf.max_epochs
        # self.sparstity_coef = (
        #     self.conf.sparstity_coef if self.conf.sparstity_coef else 0.0
        # )
        # self.max_sparstity_coef = self.conf.sparstity_coef
        # self.sparstity_coef_step = (
        #     self.conf.sparstity_coef
        #     / (total_train_steps * self.conf.sparstity_coef_warmup_duration + 1e-9)
        #     if self.conf.sparstity_coef_warmup_duration > 0
        #     else 0
        # )
        # self.step = 0
        # self.sparstity_coef_warmup = int(
        #     total_train_steps * self.conf.sparstity_coef_warmup_duration
        # )

    # ------------------------------------------------------
    #   Concept loss function for multi-concept
    # ------------------------------------------------------
    def concept_loss_fn(self, latents_pre_act, batch):
        """
        Compute the multi-concept conditioning loss by looking at each
        dimension of the latent pre-activation that corresponds to a concept.
        We apply a BCE loss with logits to each concept dimension.
        """
        total_loss = 0.0
        losses = {}

        # Toxicity concept (first latent dimension)
        mask = self.concept_masks["toxicity"](batch)
        if mask.any():
            toxicity_loss = F.binary_cross_entropy_with_logits(
                latents_pre_act[:, 0][mask],
                batch["toxicity"][mask].float(),
                reduction="mean",
            )
            total_loss += toxicity_loss * self.conf.concept_loss_weights[0]
            losses["toxicity"] = toxicity_loss.item()

        # Shakespeare concept (second latent dimension)
        mask = self.concept_masks["shakespeare"](batch)
        if mask.any():
            shakespeare_loss = F.binary_cross_entropy_with_logits(
                latents_pre_act[:, 1][mask],
                batch["label"][mask].float(),
                reduction="mean",
            )
            total_loss += shakespeare_loss * self.conf.concept_loss_weights[1]
            losses["shakespeare"] = shakespeare_loss.item()

        return total_loss, losses

    def get_combined_loss(self, latents_pre_act, latents, recons, acts, batch):
        """
        Combine reconstruction loss, sparsity loss, and multi-concept losses.
        This merges logic from the original code with the new multi-concept
        approach.
        """
        # Reconstruction loss
        l2_loss_vec = normalized_mean_squared_error(recons, acts)
        l2_loss = l2_loss_vec.mean()

        # Sparsity loss
        l1_loss_vec = normalized_L1_loss(latents, acts)
        l1_loss = l1_loss_vec.mean()

        # Multi-concept conditioning loss
        concept_loss, concept_losses = self.concept_loss_fn(latents_pre_act, batch)

        # Weighted total loss
        total_loss = l2_loss + self.conf.sparstity_coef * l1_loss + concept_loss

        # For logging, produce a dict of separate components
        loss_dict = {
            "l2": l2_loss.item(),
            "l1": l1_loss.item(),
            "concept": concept_loss.item(),
        }
        loss_dict.update(concept_losses)

        return total_loss, loss_dict

    # ------------------------------------------------------
    #   Train / Evaluate batch
    # ------------------------------------------------------

    def train_batch(self, batch: TorchData):
        """
        Exactly replicates the idea of `train_batch` under Determined,
        adapted for multi-concept usage. Returns a dict of losses.
        """
        self.model.train()
        acts = batch["acts"].to(self.device)

        # Forward pass
        latents_pre_act, latents, recons = self.model(acts)

        # Calculate loss
        total_loss, loss_dict = self.get_combined_loss(
            latents_pre_act, latents, recons, acts, batch
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update sparsity coefficient
        if self.step < self.conf.sparstity_coef_warmup_duration * len(self.dm["train"]):
            self.conf.sparstity_coef += self.sparsity_coef_step
        self.step += 1

        return loss_dict

    def evaluate_batch(self, batch: TorchData):
        """
        Exactly replicates the idea of `evaluate_batch` under Determined,
        adapted for multi-concept usage. Returns a dict of evaluation losses.
        """
        self.model.eval()
        with torch.no_grad():
            acts = batch["acts"].to(self.device)
            latents_pre_act, latents, recons = self.model(acts)
            _, loss_dict = self.get_combined_loss(
                latents_pre_act, latents, recons, acts, batch
            )
        # Prefix keys with "val_" to identify them as validation
        return {f"val_{k}": v for k, v in loss_dict.items()}

    # ------------------------------------------------------
    #   DataLoader methods
    # ------------------------------------------------------

    def build_training_data_loader(self):
        """
        Replacement for Determined's build_training_data_loader.
        Returns a DataLoader for the training subset of self.dm,
        using the configured batch size, shuffle, and num_workers=4.
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
        Returns a DataLoader for the test subset of self.dm,
        using the configured batch size, no shuffle, and num_workers=4.
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
    """
    Main entry point that parses arguments, loads a JSON config,
    instantiates SAETrial, and runs the training/validation loop.
    """
    parser = argparse.ArgumentParser(description="Train multi-concept SAE")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    conf = SAE_Train_config(**config)

    # Initialize trial
    trial = SAETrial(conf, device=args.device)
    train_loader = trial.build_training_data_loader()
    val_loader = trial.build_validation_data_loader()

    # Training loop
    for epoch in range(conf.max_epochs):
        # Training
        for batch in train_loader:
            loss_dict = trial.train_batch(batch)

            # Logging
            if trial.step % 100 == 0:
                logging.info(f"Step {trial.step}: {loss_dict}")

        # Validation
        val_losses = {}
        for batch in val_loader:
            batch_loss = trial.evaluate_batch(batch)
            for k, v in batch_loss.items():
                val_losses[k] = val_losses.get(k, 0) + v

        # Average validation losses
        avg_val = {k: v / len(val_loader) for k, v in val_losses.items()}
        logging.info(f"Epoch {epoch + 1} Validation: {avg_val}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": trial.model.state_dict(),
                "optimizer_state_dict": trial.optimizer.state_dict(),
                "config": conf.to_dict(),
            },
            f"checkpoint_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    main()
