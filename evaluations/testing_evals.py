import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import hf_hub_download
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# Additional imports for deeper evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import math


###############################################################################
# 1) Hard-coded SAE model (unchanged from your script)
###############################################################################
class SingleParamModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.weight)


class TopK(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vals, idx = torch.topk(x, self.k, dim=1)
            mask = torch.zeros_like(x).scatter_(1, idx, 1.0)
        return x * mask


class SAEModel(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

        self.pre_bias = nn.Parameter(torch.zeros(4096))  # shape [4096]
        self.encoder = SingleParamModule((12288, 4096))  # shape [12288, 4096]
        self.latent_bias = nn.Parameter(torch.zeros(12288))  # shape [12288]
        self.decoder = SingleParamModule((4096, 12288))  # shape [4096, 12288]

    def forward(self, x: torch.Tensor):
        hidden_1 = x + self.pre_bias
        hidden_1 = self.activation(hidden_1)
        hidden_2 = torch.matmul(hidden_1, self.encoder.weight.t()) + self.latent_bias
        hidden_2 = self.activation(hidden_2)
        recons = torch.matmul(hidden_2, self.decoder.weight.t())
        return hidden_1, hidden_2, recons


###############################################################################
# 2) Loading the SAE checkpoint (unchanged)
###############################################################################
def load_sae_model_from_hf(
    activation: str = "topk",
    k: int = 32,
    hf_repo_and_file="saurish/scar-multi-concept/checkpoint_epoch_100.pt",
    hf_token="",
):
    if activation.lower() == "topk":
        act_fn = TopK(k)
    elif activation.lower() == "relu":
        act_fn = nn.ReLU()
    else:
        raise ValueError(f"Unrecognized activation: {activation}")

    sae = SAEModel(act_fn)
    logging.info(f"Downloading SAE checkpoint from HF: {hf_repo_and_file}")
    if "/" in hf_repo_and_file and not hf_repo_and_file.startswith("/"):
        # Attempt HF download
        *repo_parts, ckpt_filename = hf_repo_and_file.split("/")
        repo_id = "/".join(repo_parts)
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=ckpt_filename, token=hf_token
        )
    else:
        # local path
        ckpt_path = hf_repo_and_file

    logging.info(f"Loading SAE checkpoint from {ckpt_path}")
    checkpoint_data = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in checkpoint_data:
        sd = checkpoint_data["model_state_dict"]
    elif "models_state_dict" in checkpoint_data and isinstance(
        checkpoint_data["models_state_dict"], list
    ):
        sd = checkpoint_data["models_state_dict"][0]
    else:
        sd = checkpoint_data

    sae.load_state_dict(sd, strict=True)
    sae.eval()
    return sae


###############################################################################
# 3) LLaMA loading for feed-forward activations (unchanged)
###############################################################################
def load_llama_model_for_acts(model_name_or_path):
    logging.info(f"Loading Llama model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, output_hidden_states=True
    )
    model.eval()
    return model, tokenizer


def extract_ff_acts(model, tokenizer, text, device="cuda", layer_idx=25):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    captured_acts = []

    def ff_hook(module, input_, output_):
        captured_acts.append(output_)

    hook_handle = model.model.layers[layer_idx].mlp.register_forward_hook(ff_hook)
    with torch.no_grad():
        _ = model(**inputs)
    hook_handle.remove()

    ff_out = captured_acts[0]  # shape: [batch_size, seq_len, hidden_dim]
    ff_out = ff_out[0, -1, :]  # last token of the first example
    ff_out = ff_out.half()
    return ff_out.cpu()


###############################################################################
# 4) Reconstruction & Concept detection (partly unchanged)
###############################################################################
def compare_sae_reconstruction(sae, acts, batch_size=4, device="cuda"):
    all_mses = []
    n = len(acts)
    for i in range(0, n, batch_size):
        batch_acts = acts[i : i + batch_size]
        batch_acts_tensor = torch.stack(batch_acts).to(device, dtype=torch.float16)
        with torch.no_grad():
            _, _, recons = sae(batch_acts_tensor)
            mse = ((recons - batch_acts_tensor) ** 2).mean(dim=1)
        all_mses.extend(mse.cpu().tolist())
        gc.collect()
        torch.cuda.empty_cache()
    return np.array(all_mses)


def plot_mse_hist(mse_values, out_path="sae_reconstruction_mse.png"):
    avg_mse = mse_values.mean()
    logging.info(f"Avg MSE = {avg_mse:.6f}")
    plt.figure(figsize=(7, 5))
    plt.hist(mse_values, bins=50, alpha=0.7, edgecolor="black")
    plt.title("SAE Reconstruction MSE")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.axvline(avg_mse, color="red", linestyle="--", label=f"Mean={avg_mse:.5f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    logging.info(f"Saved histogram to: {out_path}")


###############################################################################
# 5) Additional: Binning & Plotting the Concept Activation
###############################################################################
def concept_activation_vs_label_plot(
    acts,
    labels,
    sae,
    concept_idx,
    concept_name,
    num_bins=5,
    out_file="concept_vs_label.png",
):
    """
    If `labels` is continuous in [0,1], we bin them and plot the mean latent value vs. bin center.
    If `labels` is 0/1, we do a simple bar or box plot of the concept activation for each class.
    """
    device = next(sae.parameters()).device

    # Move data in batches
    B = 32
    concept_vals = []
    label_vals = []
    for i in range(0, len(acts), B):
        batch_acts = torch.stack(acts[i : i + B]).to(device, dtype=torch.float16)
        with torch.no_grad():
            _, h2, _ = sae(batch_acts)
        c = h2[:, concept_idx].detach().cpu().numpy()
        concept_vals.extend(c)
        label_vals.extend(labels[i : i + B])
    concept_vals = np.array(concept_vals)
    label_vals = np.array(label_vals)

    # Plot differently if continuous or binary
    unique_labels = np.unique(label_vals)
    plt.figure(figsize=(6, 5))

    # Heuristic: if more than 2 unique labels or if max(label) > 1.0 => treat as continuous
    if len(unique_labels) > 2 or (max(unique_labels) > 1.0):
        # We do binning
        bins = np.linspace(0, 1, num_bins + 1)  # e.g. 0..1 for toxicity
        bin_indices = np.digitize(label_vals, bins) - 1
        # compute means
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = []
        bin_stds = []
        for b_i in range(num_bins):
            mask = bin_indices == b_i
            if np.sum(mask) > 0:
                bin_means.append(np.mean(concept_vals[mask]))
                bin_stds.append(np.std(concept_vals[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        plt.errorbar(bin_centers, bin_means, yerr=bin_stds, marker="o", capsize=3)
        plt.title(f"{concept_name.capitalize()} Concept Activation vs. Label Bins")
        plt.xlabel("Label (binned)")
        plt.ylabel("Concept Neuron Activation")
    else:
        # treat as binary => boxplot or violin
        c0 = concept_vals[label_vals == 0]
        c1 = concept_vals[label_vals == 1]
        data = [c0, c1]
        plt.boxplot(data, labels=["Label=0", "Label=1"])
        plt.title(f"{concept_name.capitalize()} Concept Activation (Binary Labels)")
        plt.ylabel("Concept Neuron Activation")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Saved concept vs. label plot: {out_file}")


###############################################################################
# 6) Minimal Decision-Tree Analysis
###############################################################################
def evaluate_tree_node_count(acts, labels, sae, concept_idx, f1_threshold=0.9):
    """
    Fit a small decision tree to see how many nodes are needed to achieve `f1_threshold`.
    We try progressively deeper trees until we get the F1 or we give up at some max depth.

    For interpretability, we measure the # of nodes (or # of leaves) at the point
    where the tree achieves (or surpasses) the desired F1.
    """
    device = next(sae.parameters()).device

    # Gather concept values
    concept_vals = []
    for a in acts:
        with torch.no_grad():
            batch_a = torch.stack([a]).to(device, dtype=torch.float16)
            _, h2, _ = sae(batch_a)
            c = h2[0, concept_idx].item()
        concept_vals.append(c)

    concept_vals = np.array(concept_vals)
    label_vals = np.array(labels)

    # Now we have (concept_vals, label_vals)
    X = concept_vals.reshape(-1, 1)
    y = label_vals

    best_depth = None
    best_num_nodes = None
    max_depth_upper = 20  # arbitrary upper bound for our search

    for depth in range(1, max_depth_upper + 1):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X, y)
        preds = clf.predict(X)
        f1 = f1_score(y, preds, average="binary")  # or "macro" for multi-class
        if f1 >= f1_threshold:
            # measure number of nodes
            # for CART, #nodes = tree_.node_count
            n_nodes = clf.tree_.node_count
            best_depth = depth
            best_num_nodes = n_nodes
            break

    return best_depth, best_num_nodes


def multiple_dataset_tree_analysis(
    datasets_dict, sae, concept_indices_dict, f1_thresh=0.5
):
    """
    Example usage:
      datasets_dict = {
        "tox": (acts_list, label_list),
        "shakes": (acts_list2, label_list2)
      }
      concept_indices_dict = {
        "tox": 0,
        "shakes": 1
      }
    We iterate through each dataset, run a node-count analysis, and store results.
    Return a dictionary { dataset_name: (best_depth, best_num_nodes or None) }.
    """
    results = {}
    for ds_name, (acts, labels) in datasets_dict.items():
        c_idx = concept_indices_dict[ds_name]
        depth, n_nodes = evaluate_tree_node_count(
            acts, labels, sae, c_idx, f1_threshold=f1_thresh
        )
        results[ds_name] = (depth, n_nodes)
    return results


def plot_tree_node_counts(results_dict, out_file="tree_node_count.png"):
    """
    Takes a dictionary like:
      {
        'tox': (best_depth, best_num_nodes),
        'shakes': (best_depth, best_num_nodes)
      }
    and does a bar chart of # of nodes for each dataset (or a missing bar if None).
    """
    ds_names = list(results_dict.keys())
    node_counts = []
    for ds in ds_names:
        (_, n_nodes) = results_dict[ds]
        if n_nodes is None:
            node_counts.append(0)  # or some sentinel
        else:
            node_counts.append(n_nodes)

    plt.figure(figsize=(6, 5))
    plt.bar(ds_names, node_counts, alpha=0.7)
    plt.title("Decision Tree Node Counts to Achieve Desired F1")
    plt.ylabel("# of Tree Nodes")
    for i, val in enumerate(node_counts):
        plt.text(i, val + 0.1, f"{val}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    logging.info(f"Saved tree node count bar chart to {out_file}")


###############################################################################
# 7) OPTIONAL: Steering the LLM with scaled concept
###############################################################################
def steer_sae_latent(hidden_2, concept_idx, alpha=1.0):
    """
    Example function that scales the concept_idx dimension in hidden_2 by alpha,
    while leaving others as ReLU( ) or whatever your SAE normally does.
    In an actual generation loop, you'd:
      - run the LLM feed-forward
      - capture that as acts
      - feed acts into SAE
      - get hidden_1, hidden_2
      - then apply this function
      - decode with SAE.decoder
      - return that as the final vector to the residual.
    This is just a conceptual example if you need to produce steering outputs.
    """
    hidden_2_steered = hidden_2.clone()
    # B x D
    for i in range(hidden_2_steered.size(0)):
        # Re-apply ReLU or TopK to everything except concept_idx
        for d in range(hidden_2_steered.size(1)):
            if d == concept_idx:
                hidden_2_steered[i, d] = alpha * hidden_2_steered[i, d]
            else:
                # normally you'd do: hidden_2_steered[i,d] = ReLU(hidden_2_steered[i,d])
                hidden_2_steered[i, d] = max(0, hidden_2_steered[i, d].item())
    return hidden_2_steered


###############################################################################
# 8) MAIN DEMO
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument(
        "--sae_checkpoint",
        type=str,
        default="saurish/scar-multi-concept/checkpoint_epoch_100.pt",
    )
    parser.add_argument(
        "--activation", type=str, default="topk", choices=["topk", "relu"]
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_plot", type=str, default="sae_recon_mse.png")

    parser.add_argument("--llama_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--llama_block_idx", type=int, default=25)

    parser.add_argument("--tox_dataset", type=str, default="lmsys/toxic-chat")
    parser.add_argument("--shakes_dataset", type=str, default="artificial-shakespeare")

    parser.add_argument("--tox_concept_idx", type=int, default=0)
    parser.add_argument("--shakes_concept_idx", type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load the SAE
    sae = (
        load_sae_model_from_hf(
            activation=args.activation,
            k=args.k,
            hf_repo_and_file=args.sae_checkpoint,
            hf_token=args.hf_token,
        )
        .to(device)
        .half()
    )

    # 2) Load the Llama
    llama_model, llama_tokenizer = load_llama_model_for_acts(args.llama_model)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model.to(device).half()

    # 3) Toxic dataset
    logging.info("Loading toxicity dataset ...")
    tox_ds = load_dataset(args.tox_dataset, "toxicchat0124", split="test")
    tox_ds = tox_ds.rename_columns({"user_input": "text", "toxicity": "toxicity_score"})

    def map_text_to_acts_tox(example):
        txt = example["text"]
        acts_tensor = extract_ff_acts(
            llama_model,
            llama_tokenizer,
            txt,
            device=device,
            layer_idx=args.llama_block_idx,
        )
        example["acts"] = acts_tensor
        return example

    tox_ds = tox_ds.map(map_text_to_acts_tox, batched=False)
    tox_ds.set_format("torch", columns=["acts", "toxicity_score"])

    # Evaluate reconstruction
    all_acts_tox = [row["acts"] for row in tox_ds]
    mse_array_tox = compare_sae_reconstruction(
        sae, all_acts_tox, batch_size=args.batch_size, device=device
    )
    logging.info(
        f"[TOX] MSE: mean={mse_array_tox.mean():.4f} std={mse_array_tox.std():.4f}"
    )
    plot_mse_hist(mse_array_tox, out_path="sae_recon_toxic.png")

    # Detailed concept detection figure: bins or boxplot
    labels_tox = tox_ds["toxicity_score"]
    concept_activation_vs_label_plot(
        acts=all_acts_tox,
        labels=labels_tox,
        sae=sae,
        concept_idx=args.tox_concept_idx,
        concept_name="toxicity",
        num_bins=5,
        out_file="tox_concept_vs_label.png",
    )

    # 4) Shakespeare dataset
    if args.shakes_dataset == "artificial-shakespeare":
        art_data = [
            {
                "text": "I have a mind to strike thee ere thou speak'st .",
                "shakes_label": 1,
            },
            {"text": "Yet if thou say Antony lives, is well .", "shakes_label": 1},
            {"text": "Damn those words, 'but yet'!", "shakes_label": 0},
            {
                "text": "I have half a mind to hit you before you speak again.",
                "shakes_label": 0,
            },
        ]
        shakes_ds = Dataset.from_list(art_data)
    else:
        shakes_ds = load_dataset(args.shakes_dataset, split="test")

    def map_text_to_acts_shakes(example):
        txt = example["text"]
        acts_tensor = extract_ff_acts(
            llama_model,
            llama_tokenizer,
            txt,
            device=device,
            layer_idx=args.llama_block_idx,
        )
        example["acts"] = acts_tensor
        return example

    shakes_ds = shakes_ds.map(map_text_to_acts_shakes, batched=False)
    shakes_ds.set_format("torch", columns=["acts", "shakes_label"])

    all_acts_shakes = [row["acts"] for row in shakes_ds]
    mse_array_shakes = compare_sae_reconstruction(
        sae, all_acts_shakes, batch_size=args.batch_size, device=device
    )
    logging.info(
        f"[SHAKES] MSE: mean={mse_array_shakes.mean():.4f} std={mse_array_shakes.std():.4f}"
    )
    print("Shakespeare MSE array:", mse_array_shakes)
    plot_mse_hist(mse_array_shakes, out_path="sae_recon_shakes.png")

    labels_shakes = shakes_ds["shakes_label"]
    print("Shakespeare acts (first 5):", all_acts_shakes[:5])
    print("Shakespeare labels (first 5):", labels_shakes[:5])
    concept_activation_vs_label_plot(
        acts=all_acts_shakes,
        labels=labels_shakes,
        sae=sae,
        concept_idx=args.shakes_concept_idx,
        concept_name="shakespeare",
        out_file="shakes_concept_vs_label.png",
    )

    # 5) Minimal decision-tree isolation analysis
    # Build dictionary for each dataset: (acts_list, label_list)
    datasets_dict = {
        "tox": (all_acts_tox, labels_tox),
        "shakes": (all_acts_shakes, labels_shakes),
    }
    concept_indices_dict = {
        "tox": args.tox_concept_idx,
        "shakes": args.shakes_concept_idx,
    }
    # We aim for 90% F1 as in the paper example
    results = multiple_dataset_tree_analysis(
        datasets_dict, sae, concept_indices_dict, f1_thresh=0.5
    )
    logging.info(f"Tree analysis results: {results}")
    print("Tree analysis results:", results)
    plot_tree_node_counts(results, out_file="tree_node_count.png")

    logging.info("Done with extended concept detection & isolation analysis.")

    # 6) (Optional) Steering experiments ...
    # ... would require hooking the transformer forward pass at inference time,
    # scaling hidden_2 for the desired concept, and generating text with
    # model.generate(). You could gather the toxicity with Perspective API or
    # a local toxicity classifier, then plot alpha vs. toxicity. Because that
    # can be complex (API keys, etc.), we omit it here, but the mechanism
    # would be exactly as in the 'steer_sae_latent' function above.


if __name__ == "__main__":
    main()
