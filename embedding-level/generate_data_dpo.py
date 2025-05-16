from datasets import load_dataset

# ds_1 = load_dataset("walledai/HarmBench", "standard")
# ds_2 = load_dataset("walledai/HarmBench", "contextual")
# ds_3 = load_dataset("walledai/HarmBench", "copyright")

# print(ds_1["train"][0])
# print(ds_2["train"][0])
# print(ds_3["train"][0])

from instructions import *
from model_extraction import ModelExtraction
from classifier_manager import ClassifierManager
from model_generation import ModelGeneration
from perturbation import Perturbation
from tqdm import tqdm
import json


def get_cls_embds(model_nickname, insts):
    llm = ModelExtraction(model_nickname, device='cuda:1')

    pos_train_embds = llm.extract_embds(insts['train'][0])
    neg_train_embds = llm.extract_embds(insts['train'][1])
    pos_test_embds = llm.extract_embds(insts['test'][0])
    neg_test_embds = llm.extract_embds(insts['test'][1])

    return pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds

def get_clfr(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds):
    clfr = ClassifierManager("gen_unsafety")
    clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)

    return clfr

if __name__ == "__main__":
    # train_type = 'standard'
    # ds = load_dataset("walledai/HarmBench", train_type)
    # ds = load_dataset("walledai/AdvBench")

    model_nickname = 'llama2-7b-chat'
    dataset_name = 'Demo'

    # insts = load_instructions_by_size(
    #     dataset_name=dataset_name,
    #     label_list=["Malicious", "Safe"],
    #     train_size=0.1,
    # )
    
    # pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds = get_cls_embds(model_nickname, insts)
    # clfr = get_clfr(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)

    # pert = Perturbation(clfr, target_probability=0.05)

    llm_gen = ModelGeneration(model_nickname, device='cuda:1')
    # llm_gen.set_perturbation(pert)

    with open(f'standard_data.jsonl', 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    for idx, data in enumerate(tqdm(data)):
        output = llm_gen.generate(data['prompt'])

        new_data = {
            'prompt': data['prompt'],
            'chosen': output['completion'],
            'rejected': data['output']
        }
        
        with open(f'standard_data_dpo.jsonl', 'a') as f:
            f.write(json.dumps(new_data) + '\n')

        

