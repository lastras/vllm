# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional, Union

import huggingface_hub
import regex as re
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA, QKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LinearScalingRotaryEmbeddingWithLoRA,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLoRA,
                              QKVParallelLinearWithLoRA,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
from vllm.model_executor.layers.linear import LinearBase
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_all_lora_classes: set[type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLoRA,
}


def from_layer(layer: nn.Module,
               max_loras: int,
               lora_config: LoRAConfig,
               packed_modules_list: list,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(source_layer=layer,
                                      lora_config=lora_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config,
                                               model_config)
            return instance_layer
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(
        name: str,
        weights_mapper: Optional[WeightsMapper] = None
) -> tuple[str, bool, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
        tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
            is_bias whether the tensor is lora bias.
    """

    # LoRA weight qualified name usually starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if name.startswith("base_model.model."):
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        # recover the prefix `base_model.model.`
        name = "base_model.model." + name
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # In some situations, we may not start with `base_model.model.`.
    # If we don't (e.g., ibm-granite/granite-speech-3.3-8b),
    # we should keep the prefix intact.
    start_index = 2 if name.startswith("base_model.model.") else 0

    parts = name.split(".")
    if parts[-1] == "weight" and (parts[-2] == "lora_A"
                                  or parts[-2] == "lora_B"):
        new_name = ".".join(parts[start_index:-2])
        return new_name, parts[-2] == "lora_A", False

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "lora_embedding_A", False

    if parts[-1] == "bias":
        new_name = ".".join(parts[start_index:-2])
        return new_name, False, True

    raise ValueError(f"{name} is unsupported LoRA weight")


def is_regex_target_modules(load_modules: Union[str, list[str]],
                            expected_lora_modules: list[str]) -> bool:
    """
    PEFT supports passing `target_modules` in the form of regular expressions, 
    such as `model.*(q_proj|k_proj|v_proj)$`. This function is mainly used to 
    determine whether the suffix in the regular expression is present in the 
    `expected_lora_modules`.
    """

    def is_valid_regex(pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def is_subset(sub_list, full_list):
        return set(sub_list).issubset(set(full_list))

    # Similar to PEFT's processing logic, regex-related operations are only
    #  executed when the load_modules is a `str`.
    if not isinstance(load_modules, str):
        return False

    if is_valid_regex(load_modules):
        match = re.search(r"\((.*?)\)\$?$", load_modules)
        if match:
            suffix = match.group(1).split("|")
            return is_subset(suffix, expected_lora_modules)
    return False


def get_supported_lora_modules(model: nn.Module) -> list[str]:
    """
    In vLLM, all linear layers support LoRA.
    """
    supported_lora_modules: set[str] = set()
    # step1: traverse the model to get all the linear subfixes.
    for name, module in model.named_modules():
        if isinstance(module, (LinearBase, )):
            supported_lora_modules.add(name.split(".")[-1])
    # step 2: get the embedding modules if the model's mbedding_modules
    # is not empty.
    if model.embedding_modules:
        for name in model.embedding_modules:
            supported_lora_modules.add(name)
    return list(supported_lora_modules)


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.

    If the lora_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the lora_path as a local file path and
    converts it to an absolute path.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the lora model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith('~'):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)


    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        # the most prevalent assumption in Huggingface is that an HF repo contains at most one adapter
        # most HF model paths are of the form namespace/model, but some old paths are a model name (root level)
        # this branch of the if .. else deals with this case
        if lora_path.count("/") <= 1:
            local_snapshot_path = huggingface_hub.snapshot_download(repo_id=lora_path)
        else:
            # if the lora_path looks like namespace/model/somepath then we extract the repo_id and the subpath
            m=re.match(r"([^/]+?/[^/]+?)/(.+[^/])",lora_path)
            if m:
                repo_id = m.group(1)
                adapter_path = m.group(2)
                # we use snapshot_download to download, from the given repo, only the adapter being requested, preserving
                # the directory structure as given in the HF repo.
                local_snapshot_path = huggingface_hub.snapshot_download(repo_id=repo_id,allow_patterns=adapter_path+"/*")
                # the local path is reconstructed this way, since the convention in HF is to return the repo path regardless
                # of what allow pattern is being passed
                local_snapshot_path += "/"+adapter_path
            else:
                # exception in case the regular expression matching doesn't work.
                raise ValueError("Incorrect format for adapter path")

    except (ValueError, HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError,
            HFValidationError):
        # Handle errors that may occur during the download
        # Return original path instead instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return lora_path


    return local_snapshot_path
