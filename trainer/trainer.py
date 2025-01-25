import torch, os, json
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

from .logger import Logger
from .callbacks import EarlyStopping

from dataset_utils.cnn_dataset import CNNDataset, CNNCollatefn
from dataset_utils.utils import top_k_top_p_filtering, batch_top_k_top_p_filtering

from rouge_score import rouge_scorer

from tqdm import tqdm

class Trainer:
    def __init__(self, model:GPT2LMHeadModel,
                lang_model:str,   
                generation_kwargs:dict,        
                trainer_kwargs:dict,
                optimizer_kwargs:dict,
                lr_scheduler_kwargs:dict,
                callbacks_kwargs:dict,
                dataset_kwargs:dict):
        ''' 
        Trainer Class to fine-tune GPT2LMHeadModel on CNN Dailymail Summarization. 

        lang_model - string/name of language model. An argument necessary for creating the tokenizer(). 
        generation_kwargs - dict of generation arguments
            max_generation_length - max_length
            early_stopping - bool

        
        dataset_kwargs - 
        trainer_kwargs - dict of training arguments. 
        optimizer_kwargs - dict of arguments for initializing optimizer. 
        lr_scheduler_kwargs - dict of arguments for initializing learning rate scheduler. 
        callbacks_kwargs - dict of callback arguments. Such as patience until training termination. 

        '''
        self.model = model 
        self.device = self.model.device
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
        self.dataset_kwargs = dataset_kwargs

        self._init_cnn_dataloader(dataset_kwargs)

        self.model.resize_token_embeddings(len(self.train_dataloader.collate_fn.tokenizer))

        if os.path.exists(f'{self.output_dir}/model_checkpoints/checkpoint-model.pt'):
            state_dict = torch.load(f'{self.output_dir}/model_checkpoints/checkpoint-model.pt', map_location=torch.device(self.device))
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(
                new_state_dict
            )
            
            self.logger.log_message(f'Loaded Model checkpoint from: {self.output_dir}/model_checkpoints/checkpoint-model.pt')        

        state_dict = torch.load(f'model_ckpts/12Apr24_Run_Multi_GPU_GPT2-Medium/model_checkpoints/n_batch_ckpt.pt', map_location=torch.device(self.device))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(
            new_state_dict
        )

        self.logger.log_line()
        self.logger.log_message(f'Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Root Data Directory: {dataset_kwargs["data_dir"]}')

        self.logger.log_message(f'Train Dataset: {dataset_kwargs["train_dataset"]}')
        self.logger.log_message(f'Test Dataset: {dataset_kwargs["validation_dataset"]}')

        self.logger.log_new_line()          

        self.num_training_steps = self.total_train_batch*self.epochs
        self.num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        self.num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])        

        self.gradient_accumulation_steps = trainer_kwargs["gradient_accumulation_steps"]
        self.save_predictions = trainer_kwargs["save_predictions"]

        self.n_batch_ckpt_save = trainer_kwargs["n_batch_ckpt_save"] if "n_batch_ckpt_save" in trainer_kwargs else 0

        self.adaptive_scaler = 1.0

        self._init_optimizer(optimizer_kwargs)

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

        # put model to device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        self.logger.log_line()
        self.logger.log_message(f'Device: {self.model.device}')
        self.logger.log_new_line()                                    

        self.best_eval_loss = 10000
        self.best_eval_perplexity = 10000

        self.gpu_id = 0

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

        # self.model.resize_token_embeddings(len(self.train_dataloader.collate_fn.tokenizer))

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10

        self.ten_percent_val_batch = len(self.val_dataloader) // 10

        self.logger.log_message(f'Total Training Length: {self.total_train_batch} - ten percent log: {self.ten_percent_train_batch}')
        self.logger.log_new_line()

    def _init_optimizer(self, optimizer_kwargs:dict):
        param_dict = []

        param_dict.append({
            "params":self.model.parameters(), "lr":optimizer_kwargs["lm_lr"], "model_name":self.model.__class__.__name__
        })

        self.optimizer = getattr(
            torch.optim, optimizer_kwargs["type"]
        )(param_dict, **optimizer_kwargs["kwargs"])

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"]

        num_training_steps = self.total_train_batch*self.epochs
        num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])

        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"]) 

    def train_multi_gpu(self):
        ''' 
        Method to distribute the model training across the mentioned GPU Ids. 
        torch.nn.DataParallel() ports the model to the primary GPU and scales during runtime. 
        '''
        self.cur_epoch = 0        

        self.logger.log_line()
        self.logger.log_message(f'Training with Multi-GPU setting across GPUs : {self.multi_gpu_ids}')
        self.logger.log_new_line()
        
        self.model = torch.nn.DataParallel(self.model, device_ids=self.multi_gpu_ids)
        self.device = self.model.device_ids[-1]

        self.model.to(self.multi_gpu_ids[0])

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch

                self.train_one_epoch()

                if self.monitor_val:
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

        torch.cuda.empty_cache()
        self.model.train()
        
        train_epoch_iter = tqdm(self.train_dataloader, desc=f"Training {self.cur_epoch}")
        for batch_idx, data_items in enumerate(train_epoch_iter):
            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.device)            
                        
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(**data_items)

                del data_items

                if self.adaptive_loss_scaling:
                    loss = outputs.loss * self.adaptive_scaler

                else:
                    loss = outputs.loss        

                if loss.dim() != 0:
                    loss = torch.mean(loss)        
                
                loss = loss/self.gradient_accumulation_steps
                loss.backward()

                del outputs

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

            ''' 
            Adaptive loss Scaling strategy periodically computes the perplexity score on the 
            validation dataset. 

            The intuition of developing this strategy to prevent overfitting on the training dataset 
            and simultaneously improve generalizability of the LM. 
            '''            
            if (batch_idx + 1)%self.loss_scaling_batch == 0 and self.adaptive_loss_scaling:        
                # validation_scores = self.val_one_epoch(subset_validation=True, batch_idx=batch_idx)                
                # perf = validation_scores[self.adaptive_loss_metric]
                # delta = self.baseline_threshold - perf
                # self.adaptive_scaler = torch.sigmoid(torch.tensor(delta)).to(self.device)

                # if perf > self.baseline_threshold:
                #     self.baseline_threshold = perf

                # ''' 
                # delta +ve - sigmoid is greater than 0.5. (base_line > perf)
                # delta -ve - sigmoid is lesser than 0.5. (base_line < perf)
                # '''

                perplexity = self.eval_one_epoch()
                self.adaptive_scaler = torch.sigmoid(torch.tensor(perplexity)).to(self.device)
                
                self.logger.log_message(f'Adaptive Scaler: {self.adaptive_scaler:.4f}')

            if self.n_batch_ckpt_save:
                if (batch_idx + 1) % self.n_batch_ckpt_save == 0:
                    self.logger.log_message(f'Saving {batch_idx} Ckpt')
                    self.callbacks.save_epoch_checkpoint(self.model, path='n_batch_ckpt.pt')
            
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

        self.callbacks.save_epoch_checkpoint(self.model, f'checkpoint_{self.cur_epoch}.pt')

    def val_one_epoch(self, subset_validation:bool=False, batch_idx:int=0):

        self.model.eval()

        validation_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0} 
        total_batches = 0

        prediction_json = []

        if subset_validation:
            validation_epoch_iter = tqdm(self.val_dataloader_subset, desc=f"Computing Adaptive {self.adaptive_loss_metric} Loss Penalty - {self.cur_epoch} - {batch_idx}")
        else:
            validation_epoch_iter = tqdm(self.val_dataloader, desc=f"Validation {self.cur_epoch}")
        for batch_idx, data_items in enumerate(validation_epoch_iter):
            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.device)   

            batch_highlights = data_items["highlights"] if "highlights" in data_items else None
            batch_articles = data_items["articles"] if "articles" in data_items else None
            batch_generated_highlights = self.generate_batch_text(data_items)

            scores, batch_scores = self.compute_rogue_metrics(
                batch_generated_highlights, batch_highlights
            )

            for key in validation_scores.keys():
                validation_scores[key] += scores[key]
            
            total_batches += 1

            if subset_validation:
                continue

            if self.save_predictions and (batch_idx + 1) % self.ten_percent_val_batch == 0: 
                batch_predictions = [{"article":article, "target_highlight": target, "generated_highlight":generated} for article, target, generated in zip(batch_articles, batch_highlights, batch_generated_highlights)]
                prediction_json.extend(batch_predictions)

        for key in validation_scores.keys():
            validation_scores[key] /= total_batches  

        if subset_validation:
            return validation_scores

        else:            
            if validation_scores[self.adaptive_loss_metric] > self.baseline_threshold and self.adaptive_loss_scaling:
                self.baseline_threshold = validation_scores[self.adaptive_loss_metric]

            self.logger.log_line()
            self.logger.log_message(f'Validation Epoch: {self.cur_epoch} - Rouge Metrics: {validation_scores}')          
            self.logger.log_new_line()

            if validation_scores["rougeL"] > self.callbacks.best_score:
                self.callbacks.best_score = validation_scores["rougeL"]
                self.callbacks.save_checkpoint(self.model, self.cur_epoch)
                self.logger.log_message(f'Saving Model Checkpoint at {self.cur_epoch}')
                self.logger.log_message(f'Saving a sample of validation Model predictions')
                
                with open(f'{self.callbacks.output_dir}/prediction_json.json','w+') as f:
                    json.dump(prediction_json, f)

    def generate_batch_text(self, data_items: dict):
            
        input_ids = data_items["inference_ids"].to(self.device)
        pad_token_id = self.val_dataloader.collate_fn.tokenizer.pad_token_id

        batch_size, current_max_length = input_ids.size()
        max_length = current_max_length + self.max_generation_length

        new_input_ids = torch.full((batch_size, max_length), fill_value=pad_token_id, device=self.device)

        non_pad_mask = input_ids != pad_token_id
        last_non_pad_indices = non_pad_mask.sum(dim=1) - 1 # [indices: size = bs]

        for step in range(self.max_generation_length):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids).logits

            for idx in range(batch_size):
                last_non_pad_idx = (input_ids[idx] != pad_token_id).sum() - 1                
                ''' 
                next_token_logit : 
                -1 : log over all expect PAD token. 
                    Pros - Capture efficient context, grammer coherence
                    Cons also prone to hallucination and generate non-sensiacal info
                all : log over all vocab
                    Pros - Precise and shorter
                    Cons - Highly inaccurate and no grammer coherence
                '''
                next_token_logit = outputs[idx, last_non_pad_idx, :]/ 0.8
                # next_token_logit = outputs[idx, last_non_pad_idx, :-1]/ 0.8  # Assuming temperature of 1.0; -1 for avoiding PAD token which is last idx in vocab
                filtered_logit = top_k_top_p_filtering(next_token_logit)

                next_token = torch.multinomial(F.softmax(filtered_logit, dim=-1), num_samples=1)
                # next_step_input_ids[idx] = next_token

                new_input_ids[idx][:last_non_pad_idx + 1] = input_ids[idx][:last_non_pad_idx+1].clone() 
                new_input_ids[idx][last_non_pad_idx+1] = next_token

            if new_input_ids.size(-1) < self.val_dataloader.collate_fn.tokenizer.model_max_length:
                input_ids = new_input_ids    
            else:                
                input_ids = new_input_ids[:, self.max_generation_length:]
            
        batch_generated_highlights = []
        
        for batch_idx, input_id in enumerate(input_ids):
            last_non_pad_idx = last_non_pad_indices[batch_idx]
            generated_ids = input_id[last_non_pad_idx:].tolist() 
            text = self.val_dataloader.collate_fn.tokenizer.decode(generated_ids, skip_special_tokens=True)

            batch_generated_highlights.append(text)

        return batch_generated_highlights

    def eval_one_epoch(self, subset_validation:bool=False, batch_idx:int=0):

        self.model.eval()
        
        prediction_json = []
        eval_loss = 0.0

        if subset_validation:
            validation_epoch_iter = tqdm(self.val_dataloader_subset, desc=f"Computing Adaptive {self.adaptive_loss_metric} Loss Penalty - {self.cur_epoch} - {batch_idx}")
        else:
            validation_epoch_iter = tqdm(self.val_dataloader, desc=f"Validation {self.cur_epoch}")

        # validation_epoch_iter = tqdm(self.val_dataloader, desc=f"Validation {self.cur_epoch}")

        for batch_idx, data_items in enumerate(validation_epoch_iter):
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.device)

                batch_highlights = data_items["highlights"] if "highlights" else []
                batch_articles = data_items["articles"] if "articles" else []
                batch_inference_ids = data_items["inference_ids"] if "inference_ids" else []

                if (batch_idx + 1) % 10000 or batch_idx == 0:
                    batch_generated_highlights = self.generate_batch_text(data_items)

                    scores, batch_scores = self.compute_rogue_metrics(
                        batch_generated_highlights, batch_highlights
                    )

                    batch_predictions = [{"article":article, "target_highlight": target, "generated_highlight":generated, "rouge_scores":rouge_score} for article, target, generated, rouge_score in zip(batch_articles, batch_highlights, batch_generated_highlights, batch_scores)]
                    prediction_json.extend(batch_predictions)                    

                    with open(f'{self.callbacks.output_dir}/prediction_json.json','w+') as f:
                        json.dump(prediction_json, f)

                del data_items["highlights"]
                del data_items["articles"]
                del data_items["inference_ids"]

                with torch.no_grad(True):
                    outputs = self.model(**data_items)
                    loss = outputs.loss

                    eval_loss += loss.item()
        
        eval_loss = eval_loss/len(self.val_dataloader)
        perplexity = torch.exp(torch.tensor(eval_loss))
                
        self.logger.log_line()
        self.logger.log_message(f'Evaluation: {self.cur_epoch} - eval_loss: {eval_loss} - perplexity: {perplexity:.4f}') 
        self.logger.log_new_line()

        if self.best_eval_perplexity > perplexity:
            self.best_eval_perplexity = perplexity

            self.logger.log_message(f'Saving best-model at perplexity: {self.best_eval_perplexity}')
            self.callbacks.save_checkpoint(self.model, self.cur_epoch)

        return perplexity

        # with open(f'{self.callbacks.output_dir}/prediction_json.json','w+') as f:
        #     json.dump(prediction_json, f)

    def compute_rogue_metrics(self, generated_highlights, reference_highlights):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0} 
        batch_scores = []
               
        for gen, ref in zip(generated_highlights, reference_highlights):
            score = scorer.score(ref, gen)
            indiv_score_dict = {}
            for key in scores.keys():
                scores[key] += score[key].fmeasure
                indiv_score_dict[key] = score[key].fmeasure

            batch_scores.append(indiv_score_dict)

        # Average the scores
        for key in scores.keys():
            scores[key] /= len(generated_highlights)

        return scores, batch_scores                    
