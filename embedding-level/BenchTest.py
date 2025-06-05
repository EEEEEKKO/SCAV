from datasets import load_dataset

from instructions import *
from model_extraction import ModelExtraction
from classifier_manager import ClassifierManager
from model_generation import ModelGeneration
from perturbation import Perturbation
from tqdm import tqdm
import os
import json


def get_cls_embds(model_nickname, insts, adapter_name: str = None, experiment_name: str = None):
    llm = ModelExtraction(model_nickname, device='cuda:1', adapter_name=adapter_name, experiment_name=experiment_name)

    pos_train_embds = llm.extract_embds(insts['train'][0])
    neg_train_embds = llm.extract_embds(insts['train'][1])
    pos_test_embds = llm.extract_embds(insts['test'][0])
    neg_test_embds = llm.extract_embds(insts['test'][1])

    return pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds

def get_clfr(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds):
    clfr = ClassifierManager("test_bench")
    clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
    print(clfr.testacc)
    return clfr

if __name__ == "__main__":
    # train_type = 'standard'
    # ds = load_dataset("walledai/HarmBench", train_type)

    test_path = "harmless.jsonl"
    with open(test_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = data[:20]

    model_nickname = 'llama2-7b-chat'
    dataset_name = 'Demo'
    experiment_name = "Llama-2-7b-CAT-merged"

    insts = load_instructions_by_size(
        dataset_name=dataset_name,
        label_list=["Malicious", "Safe"],
        train_size=0.2,
    )
    
    pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds = get_cls_embds(model_nickname, insts,adapter_name=None, experiment_name=experiment_name)
    clfr = get_clfr(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)

    pert = Perturbation(clfr, target_probability=0.01)

    llm_gen = ModelGeneration(model_nickname, device='cuda:1', adapter_name=None, experiment_name=experiment_name)

    save_dir = "/user/zhaohengyu/CodeSpace/bench_results/SCAV"
    os.makedirs(save_dir, exist_ok=True)


    if "harmless" in test_path.split(".")[0]:
        llm_gen.unset_perturbation()
    elif "adv" in test_path.split(".")[0]:
        print("Setting perturbation...")
        llm_gen.set_perturbation(pert)

    with open(os.path.join(save_dir, f'{experiment_name}_{test_path.split(".")[0]}.jsonl'), 'w') as f:
        for idx, data in enumerate(tqdm(data)):

            output = llm_gen.generate(data['prompt'])

            perturbed_output = output['completion']
            print(perturbed_output.strip())

            new_data = {
                "prompt": data['prompt'],
                # "origin_output": origin_output,
                "perturbed_output": perturbed_output,
            }

            # llm_gen.unset_perturbation()
            # output = llm_gen.generate(data['prompt'])
            # print(output['completion'])
            # # input("Press Enter to continue...")

            # new_data = {
            #     "prompt": data['prompt'],
            #     # "origin_output": origin_output,
            #     "output": output['completion'],
            # }
        
            f.write(json.dumps(new_data, ensure_ascii=False) + '\n')

        

