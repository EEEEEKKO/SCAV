from model_base import ModelBase
from perturbation import Perturbation
from functools import partial
import torch
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ModelGeneration(ModelBase):
    def __init__(self, model_nickname: str, device: str = "cuda"):
        super().__init__(model_nickname, device)

        self.hooks = []
        self._register_hooks()
        self.perturbation: Perturbation = None

        self.original_outputs = []
        self.capture_original_outputs = False

        self.perturbed_outputs = []
        self.capture_perturbed_outputs = False

        self.hook_records = defaultdict(list)

    def set_perturbation(self, perturbation):
        self.perturbation = perturbation

    def _register_hooks(self):
        def _hook_fn(module, input, output, layer_idx):
            if self.capture_original_outputs:
                self.original_outputs.append(output[0].clone().detach())

            if self.perturbation is not None:
                output = self.perturbation.get_perturbation(output, layer_idx)

            if self.capture_perturbed_outputs:
                self.perturbed_outputs.append(output[0].clone().detach())

            return output
        
        for i in range(self.llm_cfg.n_layer):
            layer = self.model.model.layers[i]
            hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i))
            self.hooks.append(hook)
            
    def record_hooks(self, sample_id: str):
        """
        记录当前样本的每一层的最后一个 token 的 hook 值，并按样本ID存储。
        :param sample_id: 样本的唯一标识符（正样本或负样本的ID）
        """
        sample_hooks = []
        for i in range(self.llm_cfg.n_layer):
            if self.capture_perturbed_outputs:
                # 只记录最后一个 token 的嵌入
                sample_hooks.append(self.perturbed_outputs[i][0, -1, :].clone().detach())
            elif self.capture_original_outputs:
                # 只记录最后一个 token 的嵌入
                sample_hooks.append(self.original_outputs[i][0, -1, :].clone().detach())
        self.hook_records[sample_id] = sample_hooks
    
    def get_token_num(self,):
        return self.original_outputs[0].size(1)
    
    def check_token_embedding(self, token_id: int):
        sample_embeds = []
        for i in range(self.llm_cfg.n_layer):
            sample_embeds.append(self.original_outputs[i][0, token_id, :].clone().detach())
        return sample_embeds
    
    def get_hook_records(self):
        copied_records = {}
        for key, value in self.hook_records.items():
            copied_records[key] = [tensor.clone().detach() for tensor in value]
        return copied_records
    

    def generate(
        self, 
        prompt: str, 
        max_length: int=1024, 
        capture_perturbed_outputs: bool=True,
        capture_original_outputs: bool=True,
    ) -> dict:
        
        self.capture_original_outputs = capture_original_outputs
        self.original_outputs = []

        self.capture_perturbed_outputs = capture_perturbed_outputs
        self.perturbed_outputs = []

        prompt = self.apply_inst_template(prompt)
        input_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        input_token_number = input_ids.size(1)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            eos_token_id=terminators,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )

        result = {
            "completion_token_number": output.sequences[0].size(0) - input_token_number,
            "completion": self.tokenizer.decode(output.sequences[0][input_token_number:], skip_special_tokens=True),
        }

        def __convert(hs):
            ret = []
            for i in range(len(hs)):
                embds = torch.zeros(self.llm_cfg.n_layer, self.llm_cfg.n_dimension).to(self.device)
                for j in range(len(hs[i])):
                    embds[j, :] = hs[i][j][0, -1, :]
                ret.append(embds)
            return ret

        if self.capture_perturbed_outputs:
            n = len(self.perturbed_outputs) // self.llm_cfg.n_layer
            result["perturbed_outputs"] = __convert([self.perturbed_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        if self.capture_original_outputs:
            n = len(self.original_outputs) // self.llm_cfg.n_layer
            result["original_outputs"] = __convert([self.original_outputs[i*self.llm_cfg.n_layer:(i+1)*self.llm_cfg.n_layer] for i in range(n)])

        return result


    def __del__(self):
        for hook in self.hooks:
            hook.remove()