import os, json
import torch 

from finetune import create_teacher_model, create_student_model
from utils import create_logger, check_gpu_availability, get_gpu_count, load_from_ckpt

from trainer.trainer_minillm import PPOTrainer

def main(config_json_path:str):
    config_json = json.load(open(config_json_path))
    
    model_kwargs = config_json['model_kwargs']
    trainer_kwargs = config_json['trainer_kwargs']
    optimizer_kwargs = config_json['optimizer_kwargs']
    lr_scheduler_kwargs = config_json['lr_scheduler_kwargs']
    dataset_kwargs = config_json['dataset_kwargs']

    logger = create_logger(trainer_kwargs['output_dir'])

    is_gpu_available = check_gpu_availability()

    teacher_model = None 
    student_model = None

    if model_kwargs['teacher_lang_model'] is not None:
        teacher_model = create_teacher_model(model_kwargs)

    if model_kwargs['student_lang_model'] is not None:
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
    
    ppo_trainer = PPOTrainer(
        teacher_model=teacher_model, student_model=student_model, lang_model=model_kwargs['student_lang_model'],
        teacher_device=teacher_device, student_device=student_device,
        config_json=config_json, logger=logger
    )

    ppo_trainer.train()

    # ppo_trainer.evaluate_policy_on_train()
    
    # ppo_lm_dataloader = ppo_trainer.ppo_storage.create_loader(ppo_trainer.train_ppo_loader.batch_size)

    # for data_items in ppo_lm_dataloader:
    #     for k,v in data_items.items():
    #         if torch.is_tensor(v):
    #             print(f'{k} {v.size()}')

    #         else:
    #             print(f'{k} {len(v)}')
        
    #     print()
    #     print(data_items['query_tensors'])

    #     exit(1)


if __name__ == "__main__":

    config_json_path = "configs/config_minillm.json"

    main(config_json_path=config_json_path)