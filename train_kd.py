import json 
import torch
import os

# from model.teacher_student import TeacherStudent
from model.policy_gradient import TeacherStudent
from trainer.teacher_student_trainer import TeacherStudentTrainer
from torch.distributed import init_process_group
import torch.multiprocessing as mp

def create_model(teacher_lang_model:str, student_lang_model:str, teacher_gpu_device_id:int, student_gpu_device_id:int, generation_kwargs:dict):

    teacher_student = TeacherStudent(
        teacher_lang_model, student_lang_model, teacher_gpu_device_id, student_gpu_device_id
    )

    return teacher_student

if __name__ == "__main__":

    config = json.load(open('config_kd.json'))

    teacher_student = create_model(**config["model_arch"])

    trainer = TeacherStudentTrainer(
        teacher_student, 
        config["model_arch"]["generation_kwargs"],
        config["trainer_kwargs"], config["optimizer_kwargs"],
        config["lr_scheduler_kwargs"], config["callbacks_kwargs"],
        config["dataset_kwargs"], config["model_arch"]["teacher_lang_model"]
    )

    trainer.train()
    # trainer.train_multi_gpu()

    # if trainer.multi_gpu and len(trainer.multi_gpu_ids) != 0:
    #     trainer.train_multi_gpu()
    #     # mp.spawn(trainer.train_multi_gpu, nprocs=len(config["trainer_kwargs"]["multi_gpu_ids"]))
    # else:
    #     trainer.train()

