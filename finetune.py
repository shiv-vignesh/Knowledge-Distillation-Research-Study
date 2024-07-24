import json 
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from utils import get_gpu_count, check_gpu_availability, load_from_ckpt, create_logger
from trainer.trainer_finetune import FineTuneTrainer
from trainer.trainer_distillation import DistillationTrainer

def create_teacher_model(model_kwargs:dict):
    config = AutoConfig.from_pretrained(model_kwargs['teacher_lang_model'])
    model = AutoModelForCausalLM.from_pretrained(
        model_kwargs['teacher_lang_model'], config=config
    )

    return model

def create_student_model(model_kwargs:dict):

    config = AutoConfig.from_pretrained(model_kwargs['student_lang_model'])
    model = AutoModelForCausalLM.from_pretrained(
        model_kwargs['student_lang_model'], config=config
    )

    return model

def main(config_json_path:str):
    config_finetune = json.load(open(config_json_path))
    
    model_kwargs = config_finetune['model_kwargs']
    trainer_kwargs = config_finetune['trainer_kwargs']
    optimizer_kwargs = config_finetune['optimizer_kwargs']
    lr_scheduler_kwargs = config_finetune['lr_scheduler_kwargs']
    dataset_kwargs = config_finetune['dataset_kwargs']

    logger = create_logger(trainer_kwargs['output_dir'])

    is_gpu_available = check_gpu_availability()

    teacher_model = None 
    student_model = None

    if model_kwargs['teacher_lang_model'] is not None:
        teacher_model = create_teacher_model(model_kwargs)
    
    elif model_kwargs['student_lang_model'] is not None:
        student_model = create_student_model(model_kwargs)

    if is_gpu_available:
        ''' port student and teacher to respective gpus'''
        gpu_count = get_gpu_count()
        if gpu_count > 1:
            teacher_device_id = model_kwargs['teacher_gpu_device_id']
            student_device_id = model_kwargs['student_gpu_device_id']

            teacher_device = torch.device(f'cuda_{teacher_device_id}')
            student_device = torch.device(f'cuda_{student_device_id}')

        else:
            device = torch.device(f'cuda')

            teacher_device = device
            student_device = device

        if teacher_model is not None:
            logger.log_message('Trying to load teacher ckpt')
            teacher_model = load_from_ckpt(teacher_model, model_kwargs['teacher_model_ckpt_path'], teacher_device, logger)
        
        if student_model is not None:
            logger.log_message('Trying to load student ckpt')
            student_model = load_from_ckpt(student_model, model_kwargs['student_model_ckpt_path'], student_device, logger)

    else:
        logger.log_message(f'GPU not available; Exiting the training!!!')
        exit(1)

    if trainer_kwargs['training_type'] == "distillation":
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            lang_model=model_kwargs['student_lang_model'],
            trainer_kwargs=trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataset_kwargs=dataset_kwargs,
            logger=logger            
        )

        trainer.train()

    elif trainer_kwargs['training_type'] == "teacher_finetune":
        trainer = FineTuneTrainer(
            model=teacher_model, 
            lang_model=model_kwargs['teacher_lang_model'],
            trainer_kwargs=trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataset_kwargs=dataset_kwargs,
            logger=logger
        ) 

        trainer.train()

    elif trainer_kwargs['training_type'] == "student_finetune":
        trainer = FineTuneTrainer(
            model=student_model, 
            lang_model=model_kwargs['student_lang_model'],
            trainer_kwargs=trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataset_kwargs=dataset_kwargs,
            logger=logger
        )

        trainer.train()

if __name__ == "__main__":

    config_json_path = "configs/config_finetune.json"

    main(
        config_json_path=config_json_path
    )