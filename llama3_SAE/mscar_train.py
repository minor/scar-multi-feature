"""
Complete training script for M-SCAR, a multi-concept sparse autoencoder
hooked into a Llama-like model block.
"""

import os
import json
import math
import argparse
import logging
from typing import Optional, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from datasets import load_from_disk

hf_logging.set_verbosity_info()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

n = 2  # number of condition features


def normalized_mse_loss(
    recons: torch.Tensor, acts: Union[torch.Tensor, list]
) -> torch.Tensor:
    # Ensure acts is a tensor on the same device as recons.
    if isinstance(acts, list):
        acts = torch.stack(
            [
                a.to(recons.device)
                if isinstance(a, torch.Tensor)
                else torch.tensor(a, device=recons.device)
                for a in acts
            ],
            dim=0,
        )
    else:
        acts = acts.to(recons.device)
    numerator = (recons - acts).pow(2).sum(dim=1)
    denominator = acts.pow(2).sum(dim=1) + 1e-9
    return (numerator / denominator).mean()


def multi_feature_condition_loss(
    latents_pre_act: torch.Tensor,
    labels: Union[torch.Tensor, list],
    lambda_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Compute condition loss using the first n latent features as condition logits.
    Only valid labels (those not equal to -1) contribute to the loss.
    """
    if isinstance(labels, list):
        labels = torch.stack(
            [
                l.to(latents_pre_act.device)
                if isinstance(l, torch.Tensor)
                else torch.tensor(l, device=latents_pre_act.device)
                for l in labels
            ],
            dim=0,
        )
    else:
        labels = labels.to(latents_pre_act.device)
    # Use the first n features from the autoencoder's pre-activation outputs as condition logits.
    condition_logits = latents_pre_act[:, :n]  # expected shape: [batch_size, n]
    labels = labels.float()
    mask = labels != -1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=latents_pre_act.device)
    loss = F.binary_cross_entropy_with_logits(
        condition_logits[mask], labels[mask], reduction="sum"
    )
    loss /= mask.sum().item()
    return loss


class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = torch.topk(x, k=self.k, dim=-1)
        topk_vals = self.postact_fn(topk_vals)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk_idx, topk_vals)
        return out


class JumpReLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, threshold: torch.Tensor):
        return JumpReLUFunction.apply(x, threshold)


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return input * (input > threshold)

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_output * (input > threshold)
        grad_threshold = None
        return grad_input, grad_threshold


class Autoencoder(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_latents: int,
        activation: nn.Module = nn.ReLU(),
        tied: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.activation = activation
        self.normalize = normalize

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder = nn.Linear(n_inputs, n_latents, bias=True)
        if tied:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=True)

    def _normalize(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        std = (var + 1e-9).sqrt()
        return (x - mu) / std, mu, std

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        # x should have shape [batch_size, n_inputs]
        return self.encoder(x - self.pre_bias)

    def decode(
        self, latents: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        out = self.decoder(latents) + self.pre_bias
        if self.normalize:
            out = out * std + mu
        return out

    def forward(
        self, x: Union[torch.Tensor, list]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If x is a list, stack along dimension 0 to form a proper batch.
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = x.to(self.pre_bias.device)
        if self.normalize:
            x_normed, mu, std = self._normalize(x)
        else:
            x_normed, mu, std = x, 0, 1
        latents_pre_act = self.encode_pre_act(x_normed)
        if isinstance(self.activation, JumpReLu):
            latents = self.activation(
                latents_pre_act, torch.zeros_like(latents_pre_act)
            )
        else:
            latents = self.activation(latents_pre_act)
        recons = self.decode(latents, mu, std)
        return latents_pre_act, latents, recons


class MSCARConfig(LlamaConfig):
    model_type = "m_scar"

    def __init__(
        self,
        hook_block_num: int = 0,
        n_inputs: int = 4096,  # UPDATED: now matching the dataset's feature dimension
        n_latents: int = 4096 * 2,
        activation: str = "topk",
        activation_k: int = 32,
        tied: bool = True,
        normalize: bool = True,
        lambda_cond: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hook_block_num = hook_block_num
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.activation = activation
        self.activation_k = activation_k
        self.tied = tied
        self.normalize = normalize
        self.lambda_cond = lambda_cond


class MSCARModel(LlamaPreTrainedModel):
    config_class = MSCARConfig

    def __init__(self, config: MSCARConfig):
        super().__init__(config)
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.activation == "topk":
            activation_fn = TopK(config.activation_k, nn.ReLU())
        elif config.activation == "topk-sigmoid":
            activation_fn = TopK(config.activation_k, nn.Sigmoid())
        elif config.activation == "jumprelu":
            activation_fn = JumpReLu()
        elif config.activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unimplemented activation: {config.activation}")
        self.SAE = Autoencoder(
            n_inputs=config.n_inputs,
            n_latents=config.n_latents,
            activation=activation_fn,
            tied=config.tied,
            normalize=config.normalize,
        )
        self._sae_hook = None
        self.register_sae_hook(block_index=config.hook_block_num)
        self.post_init()

    def register_sae_hook(self, block_index: int):
        def forward_hook(module, input, output):
            bsz, seqlen, hidden_dim = output.shape
            if hidden_dim != self.config.n_inputs:
                logger.warning(
                    f"MLP hidden_size ({hidden_dim}) != n_inputs ({self.config.n_inputs}). Ensure these match or add a bridging linear layer."
                )
            x = output.reshape(bsz * seqlen, hidden_dim)
            latents_pre_act, latents, recons = self.SAE(x)
            return recons.reshape(bsz, seqlen, hidden_dim)

        self._sae_hook = self.model.layers[block_index].mlp.register_forward_hook(
            forward_hook
        )

    def remove_sae_hook(self):
        if self._sae_hook is not None:
            self._sae_hook.remove()
            self._sae_hook = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MSCARTrainer:
    def __init__(
        self,
        model: MSCARModel,
        lambda_cond: float = 1.0,
        lr: float = 1e-4,
        epochs: int = 5,
        batch_size: int = 16,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = model.config
        self.lambda_cond = lambda_cond
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        for name, param in self.model.named_parameters():
            if "SAE" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )

    def train_loop(
        self, train_dataset, val_dataset=None, label_key: str = "concept_labels"
    ):
        # --- IMPORTANT UPDATE: ensure dataset fields are in torch format ---
        if hasattr(train_dataset, "with_format"):
            train_dataset = train_dataset.with_format("torch")
        if val_dataset is not None and hasattr(val_dataset, "with_format"):
            val_dataset = val_dataset.with_format("torch")
        # --------------------------------------------------------------------

        if hasattr(train_dataset, "keys") and "train" in train_dataset:
            train_loader = DataLoader(
                train_dataset["train"], batch_size=self.batch_size, shuffle=True
            )
            if "test" in train_dataset and val_dataset is None:
                val_dataset = train_dataset["test"]
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        initial_val_mse = None
        for epoch in range(self.epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                acts = batch["acts"]
                # Only stack if acts is not already a tensor.
                if not isinstance(acts, torch.Tensor):
                    acts = torch.stack(acts, dim=0)
                if label_key in batch:
                    concept_labels = batch[label_key]
                elif "toxicity" in batch and "label" in batch:
                    concept_labels = torch.stack(
                        [batch["toxicity"], batch["label"]], dim=1
                    )
                else:
                    raise KeyError(
                        "Concept labels not found in batch. Expected key 'concept_labels' or keys 'toxicity' and 'label'."
                    )
                latents_pre_act, _, recons = self.model.SAE(acts)
                recon_loss = normalized_mse_loss(recons, acts)
                cond_loss = multi_feature_condition_loss(
                    latents_pre_act, concept_labels, lambda_mask=None
                )
                total_loss = recon_loss + self.lambda_cond * cond_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if step % 100 == 0:
                    logger.info(
                        f"Epoch {epoch} Step {step} | L_recon: {recon_loss.item():.4f}, L_cond: {cond_loss.item():.4f}, L_total: {total_loss.item():.4f}"
                    )
            if val_loader is not None:
                self.model.eval()
                val_mse_accum = 0.0
                val_cond_accum = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        for k, v in val_batch.items():
                            if isinstance(v, torch.Tensor):
                                val_batch[k] = v.to(self.device)
                        acts = val_batch["acts"]
                        if not isinstance(acts, torch.Tensor):
                            acts = torch.stack(acts, dim=0)
                        if label_key in val_batch:
                            concept_labels = val_batch[label_key]
                        elif "toxicity" in val_batch and "label" in val_batch:
                            concept_labels = torch.stack(
                                [val_batch["toxicity"], val_batch["label"]], dim=1
                            )
                        else:
                            raise KeyError(
                                "Concept labels not found in validation batch."
                            )
                        latents_pre_act, _, recons = self.model.SAE(acts)
                        rloss = normalized_mse_loss(recons, acts).item()
                        closs = multi_feature_condition_loss(
                            latents_pre_act, concept_labels, None
                        ).item()
                        val_mse_accum += rloss
                        val_cond_accum += closs
                        val_count += 1
                avg_val_mse = val_mse_accum / max(val_count, 1)
                avg_val_cond = val_cond_accum / max(val_count, 1)
                logger.info(
                    f"Epoch {epoch} Validation => MSE: {avg_val_mse:.4f}, Cond: {avg_val_cond:.4f}"
                )
                if epoch == 0:
                    initial_val_mse = avg_val_mse
        if val_loader is not None and initial_val_mse is not None:
            self.model.eval()
            val_mse_accum = 0.0
            val_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    for k, v in val_batch.items():
                        if isinstance(v, torch.Tensor):
                            val_batch[k] = v.to(self.device)
                    acts = val_batch["acts"]
                    if not isinstance(acts, torch.Tensor):
                        acts = torch.stack(acts, dim=0)
                    if label_key in val_batch:
                        concept_labels = val_batch[label_key]
                    elif "toxicity" in val_batch and "label" in val_batch:
                        concept_labels = torch.stack(
                            [val_batch["toxicity"], val_batch["label"]], dim=1
                        )
                    else:
                        raise KeyError("Concept labels not found in validation batch.")
                    latents_pre_act, _, recons = self.model.SAE(acts)
                    rloss = normalized_mse_loss(recons, acts).item()
                    val_mse_accum += rloss
                    val_count += 1
            final_val_mse = val_mse_accum / max(val_count, 1)
            delta_mse = final_val_mse - initial_val_mse
            logger.info(
                f"Final Validation MSE: {final_val_mse:.4f} | Delta MSE (final - initial): {delta_mse:.4f}"
            )

    def save_model(self, output_path: str = "m_scar_trained.pt"):
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config.to_dict(),
            },
            output_path,
        )
        logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to a torch dataset or .pt file with 'acts' and concept label keys ('concept_labels' or 'toxicity' and 'label').",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Path to a validation dataset with the same format.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_cond", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="m_scar_trained.pt")
    args = parser.parse_args()

    train_data = load_from_disk(args.train_path)
    if isinstance(train_data, dict) and "train" in train_data:
        train_dataset = train_data["train"]
    elif hasattr(train_data, "features"):
        train_dataset = train_data
    elif isinstance(train_data, list):

        class SimpleDataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                return self.data_list[idx]

        train_dataset = SimpleDataset(train_data)
    else:
        raise ValueError("Unsupported train_data format.")

    val_dataset = None
    if args.val_path is not None:
        val_data = load_from_disk(args.val_path)
        if isinstance(val_data, dict) and "train" in val_data:
            val_dataset = val_data["train"]
        elif hasattr(val_data, "features"):
            val_dataset = val_data
        elif isinstance(val_data, list):

            class SimpleDataset(Dataset):
                def __init__(self, data_list):
                    self.data_list = data_list

                def __len__(self):
                    return len(self.data_list)

                def __getitem__(self, idx):
                    return self.data_list[idx]

            val_dataset = SimpleDataset(val_data)
        else:
            raise ValueError("Unsupported val_data format.")

    print("Starting training preparation.")
    config = MSCARConfig(
        n_inputs=4096,  # UPDATED: now matching the dataset's acts dimension
        n_latents=4096 * 2,
        hook_block_num=0,
        activation="topk",
        activation_k=32,
        tied=True,
        normalize=True,
        lambda_cond=args.lambda_cond,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
    )
    model = MSCARModel(config)
    trainer = MSCARTrainer(
        model=model,
        lambda_cond=args.lambda_cond,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Starting training process.")
    trainer.train_loop(train_dataset, val_dataset)
    trainer.save_model(args.save_path)


if __name__ == "__main__":
    main()
