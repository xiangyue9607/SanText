# coding=utf-8

import logging
import math
import os
from dataclasses import field
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import os
import pickle
import time
from typing import Optional
import numpy as np
from torch.utils.data.dataset import Dataset

from filelock import FileLock

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import BatchEncoding

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from modeling_bert import BertForMaskedLM
from SanText import get_sanitized_doc

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self, inputs,
    ) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(inputs)
        inputs, labels = self.mask_tokens(*batch)
        return {"input_ids": inputs, "labels": labels}

    def _tensorize_batch(
        self, inputs: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) ->  Tuple[torch.Tensor, torch.Tensor]:
        examples = [e[0] for e in inputs]
        sanitized_examples = [e[1] for e in inputs]

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0), torch.stack(sanitized_examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return (pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id),pad_sequence(sanitized_examples, batch_first=True, padding_value=self.tokenizer.pad_token_id))

    def mask_tokens(self, labels: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
        token_embedding: np.ndarray = None,
        epsilon: float =12.0,
        threads: int = 12,

    ):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        if file_path=="Wikipedia":
            logger.info(f"Creating features from huggingface dataset file: {file_path}")
            from datasets import load_dataset
            dataset = load_dataset('wikipedia', '20200501.en')['train']
            self.examples = []
            from tqdm import tqdm
            for data in tqdm(dataset):
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data['text']))
                for i in range(0, len(tokenized_text) - block_size + 1,
                               block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
                    )
            self.sanitized_examples = get_sanitized_doc(self.examples, token_embedding, epsilon=epsilon,
                                                        threads=threads)
            logger.info("Total examples: {}. Total sanitized examples: {}", len(self.examples), len(self.sanitized_examples))

        else:
            assert os.path.isfile(file_path), f"Input file path {file_path} not found"
            directory, filename = os.path.split(file_path)

            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else directory,
                "cached_lm_{}_{}_{}".format(
                    tokenizer.__class__.__name__,
                    str(block_size),
                    filename,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    start = time.time()
                    with open(cached_features_file, "rb") as handle:
                        self.examples, self.sanitized_examples = pickle.load(handle)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )
                else:

                    logger.info(f"Creating features from dataset file at {directory}")

                    self.examples = []
                    with open(file_path, encoding="utf-8") as f:
                        text = f.read()

                    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                        self.examples.append(
                            tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
                        )
                    # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                    # If your dataset is small, first you should loook for a bigger one :-) and second you
                    # can change this behavior by adding (model specific) padding.

                    #sanitize docs
                    self.sanitized_examples=get_sanitized_doc(self.examples, token_embedding, epsilon=epsilon, threads=threads)

                    start = time.time()
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump((self.examples,self.sanitized_examples), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.sanitized_examples[i], dtype=torch.long))
