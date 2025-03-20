import os
import torch
from torch.utils.data import Dataset
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    load_from_disk,
    concatenate_datasets,
)
import transformers
from tqdm import tqdm
from create_dataset import HookedTransformer
import sys

HF_TOKEN = {}

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create datasets directory if it doesn't exist
os.makedirs("../datasets", exist_ok=True)


def get_SP_dataset():
    """
    Creates Shakespeare dataset: Modern vs Original based on data from:
    https://github.com/harsh19/Shakespearizing-Modern-English.git
    """
    ds = {
        "train": {"text": [], "label": []},
        "valid": {"text": [], "label": []},
        "test": {"text": [], "label": []},
    }

    for stage in ["train", "valid", "test"]:
        for label in ["modern", "original"]:
            filepath = f"../datasets/Shakespearizing-Modern-English/data/{stage}.{label}.nltktok"
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing file: {filepath}")

            with open(filepath, "r") as f:
                sents = [sent.strip() for sent in f.readlines()]
                labels = [label for _ in range(len(sents))]

            ds[stage]["text"] += sents
            ds[stage]["label"] += labels

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(ds["train"]),
            "valid": Dataset.from_dict(ds["valid"]),
            "test": Dataset.from_dict(ds["test"]),
        }
    )

    os.makedirs("../datasets", exist_ok=True)
    dataset.save_to_disk("../datasets/Shakespeare")
    return dataset


def create_emotion_datasets(model_type: str, block: int, site: str):
    """
    Downloads the dair-ai/emotion dataset from Hugging Face and splits it into 6 separate datasets,
    one for each emotion: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).

    Args:
        model_type (str): Model type (e.g., "llama3", "gemma2")
        block (int): Block number
        site (str): Site type (e.g., "mlp", "block")
    """
    logger.info("Downloading emotion dataset from Hugging Face...")

    # Download the dataset
    emotion_dataset = load_dataset(
        "dair-ai/emotion", "split", cache_dir="../datasets/cache"
    )

    # Process for each emotion
    emotion_labels = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise",
    }

    # For each emotion, create and save a separate dataset
    for emotion_id, emotion_name in emotion_labels.items():
        logger.info(f"Processing emotion: {emotion_name} (id: {emotion_id})")

        # Filter the dataset for this emotion
        filtered_ds = emotion_dataset.filter(
            lambda example: example["label"] == emotion_id
        )

        # Add emotion_X columns initialized with -1 for multi-dataset compatibility
        processed_ds = filtered_ds.map(
            lambda example: {
                f"emotion_{i}": 1 if i == emotion_id else 0 for i in range(6)
            }
        )

        # Add placeholder columns for toxicity and style
        processed_ds = processed_ds.map(
            lambda example: {
                "toxicity": -1,  # -1 means "no toxicity label"
                "style": -1,  # -1 means "no style label"
            }
        )

        # Process through model activations
        model_name = (
            "meta-llama/Meta-Llama-3-8B"
            if model_type == "llama3"
            else "google/gemma-2-9b"
        )

        # Create activation dataset
        logger.info(f"Generating activations for emotion: {emotion_name}")
        activation_ds = ActivationsDataset_local(
            model_name=model_name,
            hooks=block,
            site=site,
            ds_path=None,  # Using already loaded dataset
            ds_name=None,
            ds_split=None,
            ds_cache_dir="../datasets/cache_emotion",
            batch_size=64,
        )

        # Process the dataset to get activations
        processed_ds = processed_ds.map(
            lambda example: activation_ds.encode(example),
            batched=True,
            batch_size=32,
            remove_columns=[
                "text",
                "label",
            ],  # Remove original columns that will be recreated
        )

        # Save the dataset for this emotion
        output_path = (
            f"../datasets/{model_type}-emotion_{emotion_id}-B{block:02}-{site}"
        )
        logger.info(f"Saving emotion dataset to: {output_path}")
        processed_ds.save_to_disk(output_path, num_proc=32)

    logger.info("All emotion datasets created successfully!")


class ActivationsDataset(Dataset):
    """Activation Dataset of ."""

    def __init__(
        self,
        ds_path: str = "wikipedia",
        ds_name: str = "20220301.simple",
        ds_split: str = "train",
        ds_cache_dir: str = "../datasets/cache",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hooks: int = -1,
        site: str = "mlp",
        batch_size: int = 64,
        chunk_size: int = 100000,
    ) -> None:
        """
        Arguments:

        """

        self.chunk_size = chunk_size
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, token=HF_TOKEN
        )
        self.model = model.half()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, token=HF_TOKEN
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        for parameter in self.model.model.parameters():
            parameter.requires_grad = False

        torch.cuda.empty_cache()
        self.model = self.model.half()
        self.model.cuda()

        self.Hook = HookedTransformer(hooks).register_with(self.model, site)

        self.ds_path = ds_path
        self.ds_name = ds_name
        self.ds_split = ds_split
        self.ds_cache_dir = ds_cache_dir
        if ds_path == "allenai/real-toxicity-prompts":
            self.remove_columns = [
                "filename",
                "begin",
                "end",
                "challenging",
                "prompt",
                "continuation",
            ]
        self.toxic_counter = {"toxic": 0, "non-toxic": 0}
        self.loaded_dataset = self.get_dataset()
        self.iter_dataset = iter(self.loaded_dataset)
        self.samples = self.gen_samples()
        logger.info(f"Label Distribution: {self.toxic_counter}")

        self.counter: int = 0
        self.counter_limit = self.chunk_size // batch_size

    def get_dataset(self):
        dataset = load_dataset(
            self.ds_path,
            self.ds_name,
            split=self.ds_split,
            cache_dir=self.ds_cache_dir,
            streaming=True,
        )
        dataset = dataset.map(
            self.encode,
            batched=True,
            batch_size=100,
            remove_columns=self.remove_columns,
        )

        return dataset

    def encode(self, example):
        with torch.no_grad():
            toks = self.tokenizer(
                [
                    p["text"] + c["text"]
                    for p, c in zip(example["prompt"], example["continuation"])
                ],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]
            toxicity = []
            for p, c in zip(example["prompt"], example["continuation"]):
                if p["toxicity"] is not None and c["toxicity"] is not None:
                    toxicity.append((p["toxicity"] + c["toxicity"]) / 2)
                elif p["toxicity"] is None and c["toxicity"] is not None:
                    toxicity.append(c["toxicity"])
                elif p["toxicity"] is not None and c["toxicity"] is None:
                    toxicity.append(p["toxicity"])
                else:
                    toxicity.append(0)

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act, tox in zip(text, toks, acts, toxicity):
                if tox >= 0.5:
                    toxic = 1
                    self.toxic_counter["toxic"] += 1

                else:
                    toxic = 0
                    self.toxic_counter["non-toxic"] += 1

                for s, t, a in zip(sent, tok, act):
                    if s == "<|end_of_text|>":
                        break
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(toxic)

        return encoded

    def encode_rtp(self, example):
        with torch.no_grad():
            toks = self.tokenizer(
                [p["text"] for p in example["prompt"]],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]
            toxicity = []
            for p in example["prompt"]:
                if p["toxicity"] is not None:
                    toxicity.append(p["toxicity"])
                else:
                    toxicity.append(0)

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act, tox in zip(text, toks, acts, toxicity):
                if tox >= 0.5:
                    toxic = 1
                    self.toxic_counter["toxic"] += 1
                else:
                    toxic = 0
                    self.toxic_counter["non-toxic"] += 1

                for s, t, a in zip(sent, tok, act):
                    if s == "<|end_of_text|>":
                        break
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(toxic)

        return encoded

    def encode_wiki(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act in zip(text, toks, acts):
                for s, t, a in zip(sent, tok, act):
                    if s == "<|end_of_text|>":
                        break
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(-1)

        return encoded

    def encode_SP(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "label": [],
            }

            for sent, tok, act, label in zip(text, toks, acts, example["label"]):
                l = 1 if label == "original" else 0
                for s, t, a in zip(sent, tok, act):
                    if s == "<|end_of_text|>":
                        break
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["label"].append(l)

        return encoded

    def encode_CSD(self, example):
        with torch.no_grad():
            torch.cuda.empty_cache()
            toks = self.tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).input_ids

            text = [self.tokenizer.convert_ids_to_tokens(tok) for tok in toks]

            out = self.model(toks.cuda())
            acts = self.Hook.pop().detach().cpu()

            encoded = {
                "text": [],
                "input_ids": [],
                "acts": [],
                "toxicity": [],
            }

            for sent, tok, act, label in zip(text, toks, acts, example["toxicity"]):
                l = label
                for s, t, a in zip(sent, tok, act):
                    if s == "<|end_of_text|>":
                        break
                    encoded["text"].append(s)
                    encoded["input_ids"].append(t)
                    encoded["acts"].append(a)
                    encoded["toxicity"].append(l)

        return encoded

    def gen_samples(self) -> list:
        samples = None
        with torch.no_grad():
            torch.cuda.empty_cache()
        samples = []
        for _ in tqdm(range(self.chunk_size), desc="Generating token chunk"):
            try:
                samples.append(next(self.iter_dataset))
            except StopIteration:
                logger.info("Reload dataset.")
                self.iter_dataset = iter(self.get_dataset())
                samples.append(next(self.iter_dataset))

        return samples

    def __len__(self):
        return self.chunk_size

    def __getitem__(self, idx):
        new_idx = self.counter % (self.chunk_size * 10)
        if new_idx == 0 and self.counter != 0:
            self.samples = self.gen_samples()
            self.counter = 0
            return self.samples[idx]
        else:
            self.counter += 1
            return self.samples[idx]


class ActivationsDataset_local(ActivationsDataset):
    """Activation Dataset of ."""

    def __init__(
        self,
        ds_path: str = "wikipedia",
        ds_name: str = "20220301.simple",
        ds_split: str = "train",
        ds_cache_dir: str = "../datasets/cache",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hooks: int = -1,
        site: str = "mlp",
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            ds_path,
            ds_name,
            ds_split,
            ds_cache_dir,
            model_name,
            hooks,
            site,
            batch_size,
            0,
        )

        self.iter_dataset = self.loaded_dataset

    def get_dataset(self):
        if self.ds_name == "shakespeare":
            dataset_path = "../datasets/Shakespeare"

            # Check if Shakespeare dataset exists, if not, generate it
            if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                logger.info("Shakespeare dataset not found, generating...")
                get_SP_dataset()

            dataset = load_from_disk(dataset_path)
            dataset = dataset.map(
                self.encode_SP,
                batched=True,
                batch_size=25,
            )
        elif self.ds_name == "../datasets/RTP":
            dataset = load_from_disk("../datasets/RTP")
            dataset = dataset.map(
                self.encode_rtp,
                batched=True,
                batch_size=100,
                remove_columns=[
                    "filename",
                    "begin",
                    "end",
                    "challenging",
                    "prompt",
                    "continuation",
                    "sent_toxicity_bin",
                ],
            )
        elif ds_path == "nvidia/Aegis-AI-Content-Safety-Dataset-1.0":
            dataset = load_dataset(
                self.ds_path,
                cache_dir=self.ds_cache_dir,
                streaming=False,
            )
            dataset = dataset.map(
                lambda x: {
                    "toxicity": sum(
                        [
                            "Safe" != x[f"labels_{i}"]
                            for i in range(x["num_annotations"])
                        ]
                    )
                    > x["num_annotations"] / 2
                }
            )
            dataset = dataset.map(
                self.encode_CSD,
                batched=True,
                batch_size=25,
                remove_columns=[
                    "num_annotations",
                    "id",
                    "text_type",
                    "labels_0",
                    "labels_1",
                    "labels_2",
                    "labels_3",
                    "labels_4",
                ],
            )
        else:
            if self.ds_name == "wikitext-103-raw-v1":
                if "[" in self.ds_split:
                    self.ds_split, ds_split_range = self.ds_split.split("[")
                    ds_split_start, ds_split_end = ds_split_range.replace(
                        "]", ""
                    ).split(":")
                    ds_split_start = 0 if ds_split_start == "" else ds_split_start
                    logger.info(
                        f"Using subset of {self.ds_split}, with {int(ds_split_end) - int(ds_split_start)} samples."
                    )
                dataset = load_dataset(
                    self.ds_path,
                    self.ds_name,
                    split=self.ds_split,
                    cache_dir=self.ds_cache_dir,
                    streaming=False,
                )
                dataset = dataset.filter(lambda x: x["text"] != "")
                dataset = dataset.shuffle(42).select(
                    range(int(ds_split_start), int(ds_split_end))
                )
                dataset = dataset.map(
                    self.encode_wiki,
                    batched=True,
                    batch_size=25,
                )
            else:
                dataset = load_dataset(
                    self.ds_path,
                    self.ds_name,
                    split=self.ds_split,
                    cache_dir=self.ds_cache_dir,
                    streaming=False,
                )
                dataset = dataset.map(
                    self.encode,
                    batched=True,
                    batch_size=100,
                    remove_columns=self.remove_columns,
                )

        self.model = None
        self.tokenizer = None
        return dataset

    def __len__(self):
        return len(self.iter_dataset)

    def __getitem__(self, idx):
        item = self.iter_dataset[idx]
        item["acts"] = torch.tensor(item["acts"])

        return item


def create_mixture_dataset(block: int, small: bool = False):
    suff = "_small" if small else ""
    ds_wiki = load_from_disk(f"../datasets/llama3-Wiki{suff}-B{block:02}").shuffle(42)

    ds_wiki = ds_wiki.with_format("torch").train_test_split(test_size=0.1)

    ds_rtp = load_from_disk(f"../datasets/llama3-RTP-B{block:02}").shuffle(42)
    ds_rtp = ds_rtp.with_format("torch").train_test_split(test_size=0.1)

    print(ds_wiki)
    print(ds_rtp)

    dataset_train = concatenate_datasets([ds_wiki["train"], ds_rtp["train"]]).shuffle()
    dataset_test = concatenate_datasets([ds_wiki["test"], ds_rtp["test"]]).shuffle()
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    print(dataset)
    dataset.save_to_disk(
        f"../datasets/llama3-Wiki{suff}_RTP-B{block:02}",
        num_proc=32,
    )


def ensure_single_dataset(ds):
    """
    If ds is a DatasetDict, concatenate all of its splits
    into a single Dataset. If ds is already a Dataset, do nothing.
    """
    if isinstance(ds, Dataset):
        return ds
    elif isinstance(ds, DatasetDict):
        # Combine each split (train, test, val, etc.) into one big dataset
        all_splits = []
        for split_name in ds.keys():
            all_splits.append(ds[split_name])
        merged = concatenate_datasets(all_splits)
        return merged
    else:
        raise TypeError(f"Unknown dataset type: {type(ds)}")


def create_multi_concept_dataset(block: int, site="mlp"):
    rtp_path = f"../datasets/llama3-RTP-B{block:02}-{site}"
    sp_path = f"../datasets/llama3-SP-B{block:02}-{site}"
    merged_path = f"../datasets/llama3-RTP_SP-B{block:02}-{site}"

    ds_rtp = load_from_disk(rtp_path)  # Could be Dataset or DatasetDict
    ds_sp = load_from_disk(sp_path)  # Could be Dataset or DatasetDict

    # 1. Convert each to a single Dataset:
    ds_rtp = ensure_single_dataset(ds_rtp)
    ds_sp = ensure_single_dataset(ds_sp)

    # 2. Add columns to each single Dataset:
    ds_rtp = ds_rtp.add_column("label", [-1] * len(ds_rtp))  # -1 means "no style label"
    ds_sp = ds_sp.add_column(
        "toxicity", [-1] * len(ds_sp)
    )  # -1 means "no toxicity label"

    # 3. Concatenate both into one big Dataset, then do a final train_test_split
    combined = concatenate_datasets([ds_rtp, ds_sp]).shuffle(seed=42)
    final_ds = combined.train_test_split(test_size=0.1)

    # 4. Save final dataset
    final_ds.save_to_disk(merged_path)
    print(f"Saved multi-concept dataset to: {merged_path}")


# def create_multi_dataset(block: int, model_type: str, site: str):
#     """Create merged multi-concept dataset"""
#     datasets = []

#     # Load all component datasets
#     for dataset_name in ["RTP", "SP"] + [f"emotion_{i}" for i in range(6)]:
#         path = f"../datasets/{model_type}-{dataset_name}-B{block:02}-{site}"
#         ds = load_from_disk(path)

#         # Add placeholder columns if missing
#         if "toxicity" not in ds.column_names:
#             ds = ds.add_column("toxicity", [-1] * len(ds))
#         if "style" not in ds.column_names:
#             ds = ds.add_column("style", [-1] * len(ds))
#         for i in range(6):
#             col_name = f"emotion_{i}"
#             if col_name not in ds.column_names:
#                 ds = ds.add_column(col_name, [-1] * len(ds))

#         datasets.append(ds)

#     # Merge and split
#     combined = concatenate_datasets(datasets).shuffle(seed=42)
#     final_ds = combined.train_test_split(test_size=0.1)
#     final_ds.save_to_disk(f"../datasets/{model_type}-MULTI_ALL-B{block:02}-{site}")


def create_all_datasets(block: int, model_type: str, site: str):
    """
    Creates all datasets needed for training and merges them into one big dataset.

    Args:
        block (int): Block number
        model_type (str): Model type ("llama3" or "gemma2")
        site (str): Site type ("mlp" or "block")
    """
    import subprocess
    import os

    # Set up paths and model info
    model_hf = (
        "meta-llama/Meta-Llama-3-8B" if model_type == "llama3" else "google/gemma-2-9b"
    )

    logger.info("=== Starting comprehensive dataset creation process ===")

    # Step 1: Create RTP dataset
    logger.info("Step 1/5: Creating RTP dataset...")
    ds_path = "allenai/real-toxicity-prompts"
    ds_name = None
    ds_split = "train"

    rtp_dataset = ActivationsDataset_local(
        model_name=model_hf,
        hooks=block,
        site=site,
        ds_path=ds_path,
        ds_name=ds_name,
        ds_split=ds_split,
        ds_cache_dir="../datasets/cache_3",
        batch_size=2048,
    )

    rtp_dataset.iter_dataset.save_to_disk(
        f"../datasets/{model_type}-RTP-B{block:02}-{site}",
        num_proc=32,
    )
    logger.info("RTP dataset creation completed.")

    # Step 2: Clone Shakespeare repository if needed
    logger.info("Step 2/5: Setting up Shakespeare data...")
    sp_repo_path = "../datasets/Shakespearizing-Modern-English"
    if not os.path.exists(sp_repo_path):
        logger.info("Cloning Shakespeare repository...")
        result = subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/harsh19/Shakespearizing-Modern-English.git",
                sp_repo_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone repository: {result.stderr}")
            raise RuntimeError("Failed to clone Shakespeare repository")
        logger.info("Shakespeare repository cloned successfully.")
    else:
        logger.info("Shakespeare repository already exists, skipping clone.")

    # Step 3: Create SP dataset
    logger.info("Step 3/5: Creating SP dataset...")
    ds_path = "../datasets/Shakespeare"
    ds_name = "shakespeare"
    ds_split = None

    # Make sure Shakespeare dataset exists
    if not os.path.exists(ds_path) or not os.path.isdir(ds_path):
        logger.info("Shakespeare dataset not found, generating...")
        get_SP_dataset()

    sp_dataset = ActivationsDataset_local(
        model_name=model_hf,
        hooks=block,
        site=site,
        ds_path=ds_path,
        ds_name=ds_name,
        ds_split=ds_split,
        ds_cache_dir="../datasets/cache_3",
        batch_size=2048,
    )

    sp_dataset.iter_dataset.save_to_disk(
        f"../datasets/{model_type}-SP-B{block:02}-{site}",
        num_proc=32,
    )
    logger.info("SP dataset creation completed.")

    # Step 4: Create all emotion datasets
    logger.info("Step 4/5: Creating emotion datasets...")
    create_emotion_datasets(model_type, block, site)

    # Step 5: Merge all datasets
    logger.info("Step 5/5: Merging all datasets into one...")
    create_multi_dataset(block, model_type, site)

    logger.info("=== All dataset creation steps completed successfully! ===")
    logger.info(
        f"Final multi-concept dataset saved at: ../datasets/{model_type}-MULTI_ALL-B{block:02}-{site}"
    )


def create_multi_dataset(block: int, model_type: str, site: str):
    """Create merged multi-concept dataset"""
    datasets = []

    # Load all component datasets
    for dataset_name in ["RTP", "SP"] + [f"emotion_{i}" for i in range(6)]:
        path = f"../datasets/{model_type}-{dataset_name}-B{block:02}-{site}"
        ds = load_from_disk(path)

        # Add placeholder columns if missing
        if "toxicity" not in ds.column_names:
            ds = ds.add_column("toxicity", [-1] * len(ds))
        if "style" not in ds.column_names:
            ds = ds.add_column("style", [-1] * len(ds))
        for i in range(6):
            col_name = f"emotion_{i}"
            if col_name not in ds.column_names:
                ds = ds.add_column(col_name, [-1] * len(ds))

        datasets.append(ds)

    # Merge and split
    combined = concatenate_datasets(datasets).shuffle(seed=42)
    final_ds = combined.train_test_split(test_size=0.1)
    final_ds.save_to_disk(f"../datasets/{model_type}-MULTI_ALL-B{block:02}-{site}")


if __name__ == "__main__":
    if sys.argv[1] == "RTP":
        ds_path = "allenai/real-toxicity-prompts"
        ds_name = None
        ds_split = "train"
    elif sys.argv[1] == "RTP_split":
        ds_path = "../datasets/RTP"
        ds_name = "../datasets/RTP"
        ds_split = "train"
    elif sys.argv[1] == "CSD":
        ds_path = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"
        ds_name = None
        ds_split = None
    elif sys.argv[1] == "Wiki":
        ds_path = "wikitext"
        ds_name = "wikitext-103-raw-v1"
        ds_split = "train[:100000]"
    elif sys.argv[1] == "Wiki_small":
        ds_path = "wikitext"
        ds_name = "wikitext-103-raw-v1"
        ds_split = "train[:30000]"
    elif sys.argv[1] == "SP":
        ds_path = "../datasets/Shakespeare"
        ds_name = "shakespeare"
        ds_split = None
    elif sys.argv[1] == "emotion":
        # Handle emotion dataset creation
        block = int(sys.argv[2])
        model_type = sys.argv[3]  # "llama3" or "gemma2"
        site = sys.argv[4]  # "mlp" or "block"

        create_emotion_datasets(model_type, block, site)
        sys.exit(0)  # Stop here. No further dataset creation logic needed.
    elif sys.argv[1] == "multi":
        # This is where you call create_multi_concept_dataset!
        # e.g. python act_dataset.py multi 25 llama3 mlp
        block = int(sys.argv[2])
        model_type = sys.argv[3]  # "llama3"
        site = sys.argv[4]  # "mlp"

        # Actually call the merging function
        create_multi_dataset(block, model_type, site)
        sys.exit(0)  # Stop here. No further dataset creation logic needed.
    elif sys.argv[1] == "all":
        # Handle creating all datasets in one go
        block = int(sys.argv[2])
        model_type = sys.argv[3]  # "llama3" or "gemma2"
        site = sys.argv[4]  # "mlp" or "block"

        create_all_datasets(block, model_type, site)
        sys.exit(0)  # Stop here
    else:
        raise NotImplementedError

    block = int(sys.argv[2])

    if sys.argv[3] == "llama3":
        model_hf = "meta-llama/Meta-Llama-3-8B"
    elif sys.argv[3] == "gemma2":
        model_hf = "google/gemma-2-9b"
    else:
        raise NotImplementedError

    if sys.argv[4] == "mlp":
        site = "mlp"
    elif sys.argv[4] == "block":
        site = "block"
    else:
        raise NotImplementedError

    logger.info(
        f"Generating {sys.argv[1]} dataset for block {block:02} and site '{site}' of {sys.argv[3]}."
    )
    dataset = ActivationsDataset_local(
        model_name=model_hf,
        hooks=block,
        site=site,
        ds_path=ds_path,
        ds_name=ds_name,
        ds_split=ds_split,
        ds_cache_dir="../datasets/cache_3",
        batch_size=2048,
    )
    logger.info(f"Length of dataset: {len(dataset)}")

    dataset.iter_dataset.save_to_disk(
        f"../datasets/{sys.argv[3]}-{sys.argv[1]}-B{str(block) if len(str(block)) > 1 else '0' + str(block)}-{site}",
        num_proc=32,
    )
