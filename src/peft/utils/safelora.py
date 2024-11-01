# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference paper: https://arxiv.org/abs/2405.16833


import copy
import json
import os
from dataclasses import dataclass, field

import numpy
import torch
from huggingface_hub import snapshot_download
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files


@dataclass
class SafeLoraConfig:
    """
    This is the configuration class to store the configuration of a safeLoRA.
    """

    base_model_path: str = field(
        default=None,
        metadata={"help": "The path of the base model for obtaining the aligned matrix"},
    )

    aligned_model_path: str = field(
        default=None,
        metadata={"help": "The path of the aligned model for obtaining the aligned matrix"},
    )

    peft_model_path: str = field(
        default=None,
        metadata={"help": "The path of the LoRA wieghts and configs."},
    )

    select_layers_type: str = field(
        default="number",
        metadata={"help": "How to select projection layers? options: [threshold, number]"},
    )

    threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold of cosine similarity."},
    )

    num_proj_layers: int = field(
        default=10,
        metadata={"help": "The number of projected layers."},
    )

    devices: str = field(
        default="cuda",
        metadata={"help": "Devices are used in SafeLoRA. (gpu or cpu)"},
    )

    saveWeights: bool = field(
        default=True,
        metadata={"help": "Replacing and saving SafeLoRA weights to the original LoRA file."},
    )

    def __post_init__(self):
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None.")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None.")
        if self.peft_model_path is None:
            raise ValueError("peft_model_path cannot be None.")


class SafetensorLoader:
    """
    Simple utility class that loads tensors with safetensors from a single file or sharded files.

    Takes care of file name normalization etc.

    """

    def __init__(self, model_path):
        if model_path is None:
            raise ValueError("You must provide a model that model path cannot be None.")
        else:
            if os.path.exists(model_path):
                pass
            else:
                try:
                    snapshot_download(repo_id=model_path, force_download=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {str(e)}")

        suffix = "model.safetensors"
        if not model_path.endswith(suffix):
            model_path = os.path.join(model_path, suffix)

        self.model_path = model_path
        self.is_sharded = False
        self.weight_map = None

        if not os.path.exists(model_path):
            # check if the file is sharded
            par_dir = model_path.rpartition(os.path.sep)[0]
            try:
                resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                    par_dir, cached_file(par_dir, "model.safetensors.index.json")
                )
            except OSError as exc:
                raise FileNotFoundError(
                    f"Could not find file for {model_path}, ensure that there is a (sharded) safetensors file of the model."
                ) from exc

            self.is_sharded = True
            # maps from 'model-X-of-Y.safetensors' to full file path
            file_map = {k.rpartition(os.path.sep)[-1]: k for k in resolved_archive_file}
            self.weight_map = {k: file_map[v] for k, v in sharded_metadata["weight_map"].items()}

    def get_tensor(self, name):
        if not self.is_sharded:
            file_path = self.model_path
        else:
            file_path = self.weight_map[name]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            try:
                tensor = f.get_tensor(name)
            except SafetensorError as exc:
                # no matching key found, we probably need to remove the base model prefix
                if self.base_model_prefix:
                    # remove 1 extra character for "."
                    name = name[len(self.base_model_prefix) + 1 :]
                    tensor = f.get_tensor(name)
                else:
                    raise exc
        return tensor


def get_aligned_matrix(base_model_path, aligned_model_path, devices, peft_config):
    """
    Get projected matrix by following the config (target_modules) from the peft model.
    The dimensions between the base model's weights and the aligned model's weights should be the same.
    """
    sl_align = SafetensorLoader(aligned_model_path)
    sl_base = SafetensorLoader(base_model_path)

    base = [name for name in sl_base.weight_map.keys() if any(v in name for v in list(peft_config["target_modules"]))]
    align = [
        name for name in sl_align.weight_map.keys() if any(v in name for v in list(peft_config["target_modules"]))
    ]
    v = []
    for name_base, name_align in zip(base, align):
        assert (
            sl_base.get_tensor(name_base).shape == sl_align.get_tensor(name_align).shape
        ), "The dimensions of the base model's weight should be the same with the aligned model's weight."
        vec = sl_base.get_tensor(name_base) - sl_align.get_tensor(name_align)
        vec = vec.to(devices)
        if devices == "cpu":
            vec = vec.to(torch.float32)
        vec = torch.mm(vec, vec.t()) / torch.norm(vec)
        v.append((vec).detach().cpu())
    return v


def project_weights(configs, peft_weights, v):
    ori_peft_weights = copy.deepcopy(peft_weights)
    names_A = [name for name in peft_weights.keys() if "lora_A" in name]
    names_B = [name for name in peft_weights.keys() if "lora_B" in name]
    idx = 0
    i = 0
    dis = []
    cos_total = []
    for name_A, name_B in zip(names_A, names_B):
        A = ori_peft_weights[name_A]
        if configs.devices != "cpu":
            P = v[idx].to(torch.bfloat16).to(configs.devices)
        else:
            P = v[idx].to("cpu")
        W = torch.mm(P, ori_peft_weights[name_B])
        fW = torch.mm(W, A)
        ori = torch.mm(ori_peft_weights[name_B], A)

        cos = numpy.round(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item(), 5)
        cos_total.append(cos)
        if cos <= configs.threshold:
            i += 1
            peft_weights[name_B] = W
        else:
            peft_weights[name_B] = ori_peft_weights[name_B]

        dist = 1 / (1 + torch.norm(peft_weights[name_B].reshape(1, -1) - W.reshape(1, -1)))

        dis.append(dist.item())
        idx += 1
    return peft_weights, cos_total


def apply_safelora(configs):
    """
    ===================================================
    An example of how to use apply_safelora() function
    ===================================================

    config = SafeLoRAConfig(base_model_path='../LLM_Models/llama-2-7b-hf/',\
                            aligned_model_path='../LLM_Models/llama-2-7b-chat-fp16/',
                            peft_model_path = '../finetuneLLM/finetuned_models/samsumBad-7b-fp16-peft-seed-42',
                            devices='cuda',
                            select_layers_type='threshold',
                            saveWeights=True)

    final_lora_weight = apply_safelora(config)

    If config.saveWeights is True, the original LoRA weight file will be replaced by the SafeLoRA weights.
    """

    with open(f"{os.path.join(configs.peft_model_path, 'adapter_config.json')}") as f:
        peft_config = json.load(f)
    v = get_aligned_matrix(configs.base_model_path, configs.aligned_model_path, configs.devices, peft_config)
    with safe_open(
        f"{os.path.join(configs.peft_model_path, 'adapter_model.safetensors')}", framework="pt", device=configs.devices
    ) as f:
        if configs.devices == "cpu":
            peft_weights = {name: f.get_tensor(name).to(torch.float32) for name in f.keys()}
        else:
            peft_weights = {name: f.get_tensor(name).to(torch.bfloat16) for name in f.keys()}
    if configs.select_layers_type == "threshold":
        final_wieghts, _ = project_weights(configs, peft_weights, v)
    elif configs.select_layers_type == "number":
        _, cos = project_weights(configs, peft_weights, v)
        thrs = numpy.sort(cos)[: configs.num_proj_layers][-1]
        configs.threshold = thrs
        final_wieghts, _ = project_weights(configs, peft_weights, v)

    if configs.saveWeights:
        save_file(final_wieghts, f"{os.path.join(configs.peft_model_path, 'adapter_model.safetensors')}")

    return final_wieghts
