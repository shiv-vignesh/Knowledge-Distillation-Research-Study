import json 
import torch
import os

from transformers import GPT2LMHeadModel
from trainer.trainer import Trainer

def create_model(model_arch:dict, device_id:int):
    ''' 
    Method to Create a GPT2LMHeadModel. 

    model_arch['lang_model'] : gpt2 or gpt2-medium.
    device_id : GPU Device ID,
    '''    
    if "gpt2" in model_arch["lang_model"]:
        model = GPT2LMHeadModel.from_pretrained(model_arch['lang_model'])

    if device_id is not None:
        device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device(f"cuda") if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model

if __name__ == "__main__":

    config = json.load(open('config.json'))
    model = create_model(
        config["model_arch"], device_id=config["trainer_kwargs"]["multi_gpu_ids"][0]
    )

    trainer = Trainer(
        model, config["model_arch"]["lang_model"],
        config["model_arch"]["generation_kwargs"],
        config["trainer_kwargs"], config["optimizer_kwargs"],
        config["lr_scheduler_kwargs"], config["callbacks_kwargs"],
        config["dataset_kwargs"]
    )

    if trainer.multi_gpu and len(trainer.multi_gpu_ids) != 0:
        trainer.train_multi_gpu()
    else:
        trainer.train()

