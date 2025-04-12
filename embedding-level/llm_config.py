__cfg = {
    'llama2-7b-chat': {
        'model_nickname': 'llama2-7b-chat',
        'model_name': 'meta-llama/Llama-2-7b-chat-hf', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
    'llama2-7b-base':{
        'model_nickname': 'llama2-7b-base',
        'model_name': 'meta-llama/Llama-2-7b-hf', 
        'n_layer': 32, 
        'n_dimension': 4096
    },
    'llama3-8b-instruct': {
        'model_nickname': 'llama3-8b-instruct',
        'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 
        'n_layer': 32, 
        'n_dimension': 4096
    }, 
}

class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)

def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])