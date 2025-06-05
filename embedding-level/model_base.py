from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_config import cfg, get_cfg
import torch
import os

class ModelBase:
    def __init__(self, model_nickname: str, device: str = "cuda", adapter_name: str = None, experiment_name: str = None):
        self.llm_cfg = get_cfg(model_nickname)
        self.device = device
        self.experiment_name = experiment_name
        
        if experiment_name == "meta-llama":
            local_model_dir = f"/user/zhaohengyu/CodeSpace/models/meta-llama/Llama-2-7b-chat-hf"
        elif experiment_name == "Llama-2-7b-CAT-merged":
            local_model_dir = f"/user/zhaohengyu/CodeSpace/models/ContinuousAT/Llama-2-7B-CAT-merged"
        else:
            local_model_dir = f"/user/zhaohengyu/CodeSpace/models/our_models/{self.experiment_name}"
        print(f"Loading model from {local_model_dir}...")

        # 检查本地是否已有模型和分词器
        if not os.path.exists(local_model_dir):
            print(f"模型 {self.llm_cfg.model_name} 未在本地找到，正在下载...")
            os.makedirs(local_model_dir, exist_ok=True)
            # 下载并保存模型和分词器到本地
            model = AutoModelForCausalLM.from_pretrained(self.llm_cfg.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.model_name)
            model.save_pretrained(local_model_dir)
            tokenizer.save_pretrained(local_model_dir)
            print(f"模型 {self.llm_cfg.model_name} 已保存到本地：{local_model_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(local_model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        if adapter_name is not None:
            print(f"Loading adapter {adapter_name}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_name).to(self.device)
        else:
            self.model = base_model

    def apply_sft_template(self, instruction, system_message=None):
        if system_message is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
        return messages
    
    def apply_inst_template(self, text):
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        return messages