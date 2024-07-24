import torch
from torch.utils.data import DataLoader, DistributedSampler

from .logger import Logger
from .losses import Losses
from dataset_utils.cnn_dataset import CNNDataset, CNNCollatefn

from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm

class DistillationTrainer:
    def __init__(self, teacher_model:torch.nn.Module, student_model:torch.nn.Module, 
                 lang_model:str, trainer_kwargs:dict, optimizer_kwargs:dict, 
                 lr_scheduler_kwargs:dict, dataset_kwargs:dict,
                 logger:Logger):   
        
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.lang_model = lang_model
        
        self.logger = logger

        self.output_dir = trainer_kwargs['output_dir']
        self.epochs = trainer_kwargs['epochs']
        self.monitor_train = trainer_kwargs['monitor_train']
        self.monitor_val = trainer_kwargs['monitor_val']
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        self.kd_ratio = trainer_kwargs['kd_ratio']

        '''#TODO, Multi-GPU Flag'''

        self.logger.log_line()
        self.logger.log_message(f'Initializing Dataloaders')
        self._init_dataloader(dataset_kwargs)

        if "GPT2" in self.teacher_model.__class__.__name__:
            self.teacher_model.resize_token_embeddings(
                len(self.train_dataloader.collate_fn.tokenizer)
            )

        if "GPT2" in self.student_model.__class__.__name__:
            self.student_model.resize_token_embeddings(
                len(self.train_dataloader.collate_fn.tokenizer)
            )

        self.logger.log_message(f'Adding {self.train_dataloader.collate_fn.highlight_start_token}')

        self.train_dataloader.collate_fn.tokenizer.add_special_tokens(
            {'additional_special_tokens':["<highlight_start>", "<highlight_end>"]}
        )

        self.validation_dataloader.collate_fn.tokenizer.add_special_tokens(
            {'additional_special_tokens':["<highlight_start>", "<highlight_end>"]}
        )

        self.teacher_model.resize_token_embeddings(
            len(self.train_dataloader.collate_fn.tokenizer)
        )

        self.student_model.resize_token_embeddings(
            len(self.train_dataloader.collate_fn.tokenizer)
        )

        self.logger.log_line()
        self.logger.log_message(f'Initializing optimizer')
        self.logger.log_new_line()
        
        self._init_optimizer(optimizer_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        
        self.logger.log_new_line()

        self.num_training_steps = self.total_train_batch*self.epochs
        self.num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        self.num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])        

        self.gradient_accumulation_steps = trainer_kwargs["gradient_accumulation_steps"]

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line()

        self.logger.log_message(f'Initializing Learning Rate Scheduler')
        self._init_lr_scheduler(lr_scheduler_kwargs)

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line() 

        ''' 
        #TODO, Callbacks Initialization
        '''

        self.teacher_model_device = self.teacher_model.device
        self.student_model_device = self.student_model.device

    def _init_optimizer(self, optimizer_kwargs:dict):
        param_dict = []

        param_dict.append({
            "params":self.student_model.parameters(), "lr":optimizer_kwargs["lm_lr"], "model_name":self.student_model.__class__.__name__
        })

        self.optimizer = getattr(
            torch.optim, optimizer_kwargs['type']
        )(param_dict, **optimizer_kwargs['kwargs'])

    def _init_lr_scheduler(self, lr_scheduler_kwargs):

        if lr_scheduler_kwargs['_type'] == "linear_lr_warmup":
            num_warmup_steps = lr_scheduler_kwargs['num_warmup_steps']
            num_training_steps = lr_scheduler_kwargs['num_training_steps']

            num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
            num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])

            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                                num_warmup_steps=num_warmup_steps,
                                                                num_training_steps=num_training_steps)
            
        elif lr_scheduler_kwargs['_type'] == "polynomial_lr_warmup":
            num_warmup_steps = lr_scheduler_kwargs['num_warmup_steps']
            num_training_steps = lr_scheduler_kwargs['num_training_steps']

            self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, 
                                                                          num_warmup_steps=num_warmup_steps,
                                                                          num_training_steps=num_training_steps
                                                                          )

    def _init_dataloader(self, dataset_kwargs:dict, multi_gpu:bool=False):        
        def _init_cnn_dataloader_helper(root_dir:str, csv_file:str, batch_size:int, dataset_type:str):
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

        self.dataset_type = dataset_kwargs['_type']

        self.logger.log_line()
        self.logger.log_message(f'Dataset for Training {self.dataset_type}')
        self.logger.log_new_line()

        if self.dataset_type == "cnn_dailymail":

            self.logger.log_message(f'Dataset Type: {self.dataset_type}')

            cnn_dataset_kwargs = dataset_kwargs['cnn_dataset_kwargs']
            self.train_dataloader = _init_cnn_dataloader_helper(
                cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['train_dataset'],
                cnn_dataset_kwargs['train_batch_size'], dataset_type='train'
            )

            self.validation_dataloader = _init_cnn_dataloader_helper(
                cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['validation_dataset'],
                cnn_dataset_kwargs['val_batch_size'], dataset_type='validation'                
            )

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10

        self.ten_percent_val_batch = len(self.validation_dataloader) // 10

        self.logger.log_message(f'Total Training Length: {self.total_train_batch} - ten percent log: {self.ten_percent_train_batch}')
        self.logger.log_new_line()

    def create_mask(self, labels:torch.tensor):

        highlight_start_id = self.train_dataloader.collate_fn.tokenizer.convert_tokens_to_ids("<highlight_start>")
        highlight_end_id = self.train_dataloader.collate_fn.tokenizer.convert_tokens_to_ids("<highlight_end>")

        highlight_start_index = (labels == highlight_start_id).nonzero(as_tuple=True)[1]
        highlight_end_index = (labels == highlight_end_id).nonzero(as_tuple=True)[1]

        mask = torch.zeros_like(labels)

        for batch_idx, (start_idx, end_idx) in enumerate(zip(highlight_start_index, highlight_end_index)):
            mask[batch_idx, start_idx:end_idx] = 1

        return mask

    def train(self):
        self.cur_epoch = 0        

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch
                self.train_one_epoch()

                if self.monitor_val:
                    # self.val_one_epoch()
                    '''TODO, Fix eval_one_epoch() method'''
                    self.eval_one_epoch()

        except KeyboardInterrupt:
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            exit(1)   

    def train_one_epoch(self):

        self.teacher_model.eval()
        self.student_model.train()

        total_loss = 0.0
        ten_per_batch_loss = 0.0

        torch.cuda.empty_cache()

        train_epoch_iter = tqdm(self.train_dataloader, desc=f"Training {self.cur_epoch}")
        for batch_idx, data_items in enumerate(train_epoch_iter):
            _ids = data_items['_ids']        
            del data_items['_ids']

            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.teacher_model_device)            
                        
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    **data_items
                ).logits

            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.student_model_device)            

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                student_logits = self.model(**data_items).logits
            
            mask = self.create_mask(data_items['labels'])

            #rightShift - Autoregressive 
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            shift_student_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = data_items['labels'][..., 1:].contiguous()
            shift_mask = mask[..., 1:].contiguous()

            loss = Losses.cross_entropy_loss(shift_student_logits, shift_labels, shift_mask)

            if self.kd_ratio:
                kd_loss = Losses.forward_kl(
                    shift_teacher_logits, shift_student_logits, shift_mask
                )            

                loss = (1 - self.kd_ratio) * loss + self.kd_ratio * kd_loss

            loss = loss/self.gradient_accumulation_steps if self.gradient_accumulation_steps != 0 else loss
            loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            if not self.gradient_accumulation_steps:
                self.optimizer.step()
                self.lr_scheduler.step()                  
            
            elif (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()

            total_loss += loss.item()
            ten_per_batch_loss += loss.item()

            if (batch_idx + 1)%self.ten_percent_train_batch == 0:
                average_loss = ten_per_batch_loss/self.ten_percent_train_batch
                perplexity = torch.exp(torch.tensor(average_loss))
                self.logger.log_message(f'Epoch: {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - loss:{average_loss:.4f} - perplexity: {perplexity:.4f}')

                ten_per_batch_loss = 0.0


        average_loss = total_loss/len(self.train_dataloader)
        perplexity = torch.exp(torch.tensor(average_loss))
        self.logger.log_line()
        self.logger.log_message(f'Training Epoch: {self.cur_epoch} - loss: {average_loss:.4f} perplexity: {perplexity:.4f}')
        self.logger.log_new_line()


            
                        


