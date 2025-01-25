import torch, os, json
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from .logger import Logger
from .callbacks import EarlyStopping

from dataset_utils.cnn_dataset import CNNDataset, CNNCollatefn
from dataset_utils.utils import top_k_top_p_filtering, batch_top_k_top_p_filtering
from model.teacher_student import TeacherStudent

from rouge_score import rouge_scorer
from tqdm import tqdm

class TeacherStudentTrainer:

    def __init__(self, 
                teacher_student:TeacherStudent,  
                generation_kwargs:dict,        
                trainer_kwargs:dict,
                optimizer_kwargs:dict,
                lr_scheduler_kwargs:dict,
                callbacks_kwargs:dict,
                dataset_kwargs:dict,
                lang_model:str  
                 ):
        
        self.teacher_student = teacher_student

        self.teacher_device = self.teacher_student.teacher_model.device
        self.student_device = self.teacher_student.student_model.device

        self.lang_model = lang_model

        self.epochs = trainer_kwargs["epochs"]
        self.output_dir = trainer_kwargs["output_dir"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]

        self.max_generation_length = generation_kwargs["max_generation_length"]
        self.num_beams = generation_kwargs["num_beams"]
        self.generation_early_stopping = generation_kwargs["early_stopping"]
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  

        self.logger = Logger(trainer_kwargs)

        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]

        self.metrics = trainer_kwargs["metrics"]

        self.adaptive_loss_scaling = trainer_kwargs["adaptive_loss_scaling"]
        self.loss_scaling_batch = trainer_kwargs["loss_scaling_batch"] if trainer_kwargs["loss_scaling_batch"] != -1 else 1024
        self.adaptive_loss_metric = trainer_kwargs["adaptive_loss_metric"] if trainer_kwargs["adaptive_loss_metric"] != -1 else "rougeL"
        self.baseline_threshold = trainer_kwargs["baseline_threshold"] if trainer_kwargs["baseline_threshold"] != -1 else 0.15

        self.multi_gpu = trainer_kwargs["multi_gpu"] if "multi_gpu" in trainer_kwargs else False
        self.multi_gpu_ids = trainer_kwargs["multi_gpu_ids"] if "multi_gpu_ids" in trainer_kwargs else []

        self._init_cnn_dataloader(dataset_kwargs)

        if os.path.exists(trainer_kwargs["teacher_ckpt_path"]):
            self.logger.log_message(f'Loading Teacher Checkpoint Model: {trainer_kwargs["teacher_ckpt_path"]}')
            self.teacher_student.load_teacher_ckpt(
                trainer_kwargs["teacher_ckpt_path"],
                len(self.train_dataloader.collate_fn.tokenizer)
            )

        if os.path.exists(trainer_kwargs["student_ckpt_path"]):
            self.logger.log_message(f'Resuming/Loading Student Checkpoint Model: {trainer_kwargs["student_ckpt_path"]}')
            self.teacher_student.load_student_ckpt(
                trainer_kwargs["student_ckpt_path"],
                len(self.train_dataloader.collate_fn.tokenizer)
            )

        # self.teacher_student = self.port_to_peft(self.teacher_student)
        
        self.gradient_accumulation_steps = trainer_kwargs["gradient_accumulation_steps"]
        self.num_training_steps = self.total_train_batch*self.epochs            
        self.num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        self._init_optimizer(optimizer_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Teacher Device: {self.teacher_student.teacher_model.device}')
        self.logger.log_new_line()                                    

        self.logger.log_line()
        self.logger.log_message(f'Student Device: {self.teacher_student.student_model.device}')
        self.logger.log_new_line()                

        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()        

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line() 

        self._init_lr_scheduler(lr_scheduler_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
        self.logger.log_new_line()
        for k, v in self.lr_scheduler.state_dict().items():
            self.logger.log_message("{:<30} {}".format(k, v))

        self._init_callbacks(callbacks_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Callbacks: {self.callbacks.__class__.__name__}')
        self.logger.log_new_line()
        self.logger.log_message("{:<30} {}".format('save_final_model', self.callbacks.save_final_model))
        self.logger.log_message("{:<30} {}".format('patience', self.callbacks.patience))
        self.logger.log_message("{:<30} {}".format('threshold', self.callbacks.threshold))
        self.logger.log_message("{:<30} {}".format('mode', self.callbacks.mode))

        self.best_eval_loss = 10000
        self.best_eval_perplexity = 10000

        self.gpu_id = 0
        self.n_batch_ckpt_save = trainer_kwargs["n_batch_ckpt_save"] if "n_batch_ckpt_save" in trainer_kwargs else 0

    def _init_cnn_dataloader(self, dataset_kwargs:dict, multi_gpu:bool=False):
        def _init_dataloader_helper(root_dir:str, csv_file:str, batch_size:int, dataset_type:str):
            dataset = CNNDataset(
                f'{root_dir}/{csv_file}',dataset_type
            )

            if not multi_gpu:
                dataloader = DataLoader(
                    dataset, batch_size=batch_size,
                    collate_fn=CNNCollatefn(
                        self.lang_model,
                        dataset_type=dataset_type
                    ), 
                    shuffle=True,
                    pin_memory=True
                )

                return dataloader
            
            else:
                dataloader = DataLoader(
                    dataset, batch_size=batch_size,
                    collate_fn=CNNCollatefn(
                        self.lang_model,
                        dataset_type=dataset_type
                    ), 
                    shuffle=True,
                    sampler=DistributedSampler(dataset),
                    pin_memory=True
                )

                return dataloader

        self.train_dataloader = _init_dataloader_helper(
            dataset_kwargs['data_dir'], dataset_kwargs['train_dataset'], dataset_kwargs['train_batch_size'], dataset_type='train'
        )

        self.val_dataloader = _init_dataloader_helper(
            dataset_kwargs['data_dir'], dataset_kwargs['validation_dataset'], dataset_kwargs['val_batch_size'], dataset_type='validation'
        )

        if self.adaptive_loss_scaling:
            if dataset_kwargs["validation_subset"]:
                self.val_dataloader_subset = _init_dataloader_helper(
                    dataset_kwargs['data_dir'], dataset_kwargs['validation_subset'], dataset_kwargs['val_batch_size'], dataset_type='validation'
                )

            else:
                self.logger.log_message(f'Adaptive Loss Scaling is set to true - validation subset not mentioned in config')
                self.adaptive_loss_scaling = False

        else:
            self.val_dataloader_subset = None

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 20

        self.ten_percent_val_batch = len(self.val_dataloader) // 10

        self.logger.log_message(f'Total Training Length: {self.total_train_batch} - ten percent log: {self.ten_percent_train_batch}')
        self.logger.log_new_line()

        self.teacher_student.student_model.resize_token_embeddings(
            len(self.train_dataloader.collate_fn.tokenizer)
        )

        self.teacher_student.teacher_model.resize_token_embeddings(
            len(self.train_dataloader.collate_fn.tokenizer)
        )        

    def _init_optimizer(self, optimizer_kwargs: dict):
        param_dict = []

        param_dict.append({
            "params":self.teacher_student.student_model.parameters(), "lr":optimizer_kwargs["lm_lr"], "model_name":self.teacher_student.student_model.__class__.__name__
        })

        self.optimizer = getattr(
            torch.optim, optimizer_kwargs["type"]
        )(param_dict, **optimizer_kwargs["kwargs"])

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"]

        num_training_steps = self.total_train_batch*self.epochs
        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.num_training_steps)

    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"]) 

    def train_multi_gpu(self):
        '''
        TODO- with DistributedDataParallel
        '''
        self.cur_epoch = 0        

        self.logger.log_line()
        self.logger.log_message(f'Training with Multi-GPU setting across GPUs : {self.multi_gpu_ids}')
        self.logger.log_new_line()
        
        self.teacher_student.student_model = torch.nn.DataParallel(self.teacher_student.student_model, device_ids=self.multi_gpu_ids[1:])
        self.teacher_student.teacher_device = self.multi_gpu_ids[0]
        self.teacher_student.student_device = self.multi_gpu_ids[1]

        self.teacher_student.teacher_model.to(self.multi_gpu_ids[0])
        self.teacher_student.student_model.to(self.multi_gpu_ids[1])

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch

                self.train_one_epoch()

                if self.monitor_val:
                    # self.val_one_epoch()
                    self.eval_one_epoch()

        except KeyboardInterrupt:
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            exit(1)    
                    
    def train(self):
        self.cur_epoch = 0     

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch

                self.train_one_epoch()

                if self.monitor_val:
                    # self.val_one_epoch()
                    self.eval_one_epoch()

        except KeyboardInterrupt:
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            exit(1)                        


    def train_one_epoch(self):

        total_loss = 0.0
        ten_per_batch_loss = 0.0

        self.teacher_student.student_model.train()
        self.teacher_student.teacher_model.eval()

        train_epoch_iter = tqdm(self.train_dataloader, desc=f"Training {self.cur_epoch}")
        for batch_idx, data_items in enumerate(train_epoch_iter):

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                loss = self.teacher_student(
                    data_items["input_ids"],
                    data_items["attention_mask"],
                    data_items["labels"]
                )

                if loss.dim() != 0:
                    loss = torch.mean(loss)        
                
                loss = loss/self.gradient_accumulation_steps
                loss.backward()
                # scaler.scale(loss).backward()

                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.teacher_student.student_model.parameters(), self.gradient_clipping)
                
                if not self.gradient_accumulation_steps:
                    # scaler.step(self.optimizer)
                    # scaler.update()
                    self.optimizer.step()
                    self.lr_scheduler.step()                  
                
                elif (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # scaler.step(self.optimizer)
                    # scaler.update()

                    self.optimizer.step()
                    self.lr_scheduler.step()

            total_loss += loss.item()
            ten_per_batch_loss += loss.item()

            if (batch_idx + 1)%self.ten_percent_train_batch == 0:
                average_loss = ten_per_batch_loss/self.ten_percent_train_batch
                perplexity = torch.exp(torch.tensor(average_loss))
                self.logger.log_message(f'Epoch: {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - loss:{average_loss:.4f} - perplexity: {perplexity:.4f}')

                ten_per_batch_loss = 0.0

            if self.n_batch_ckpt_save:
                if (batch_idx + 1) % self.n_batch_ckpt_save == 0:
                    self.logger.log_message(f'Saving {batch_idx} Ckpt')
                    self.callbacks.save_epoch_checkpoint(self.teacher_student.student_model, path='n_batch_ckpt.pt')
            
            if batch_idx > 65000:
                self.logger.log_line()
                self.logger.log_message(f'Reached Maximum number of training steps. Exiting Epoch Training')
                self.logger.log_new_line()
                break

        average_loss = total_loss/len(self.train_dataloader)
        perplexity = torch.exp(torch.tensor(average_loss))
        self.logger.log_line()
        self.logger.log_message(f'Training Epoch: {self.cur_epoch} - loss: {average_loss:.4f} perplexity: {perplexity:.4f}')
        self.logger.log_new_line()

        self.callbacks.save_epoch_checkpoint(self.teacher_student.student_model, f'checkpoint_{self.cur_epoch}.pt')
