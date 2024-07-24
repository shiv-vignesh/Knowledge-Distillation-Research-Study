import os, gc
import torch
import torch.utils
import torch.utils.data
from transformers import AutoModelForCausalLM, GenerationConfig, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from tqdm import tqdm

from dataset_utils.cnn_dataset import PPO_CNNCollatefn, CNNDataset
from dataset_utils.data_types import PPORLElement, PPORLBatch
from dataset_utils.storages import PPOStorage

from .losses import Losses
from .logger import Logger

''' 
train_ppo_loader 
- Created from PPOCollateFn()
- Evaluate model's policy, collect rewards and store in PPOStorage. 

ppo_lm_dataloader
- Created from PPOStorage()
- Sample PPORLElements/PPORLBatch during policy finetuning (train)

'''

class PPOTrainer:

    def __init__(self, teacher_model:AutoModelForCausalLM, student_model:AutoModelForCausalLM, 
                 lang_model:str, teacher_device:torch.device, student_device:torch.device,
                 config_json:dict, logger:Logger):

        self.teacher_model = teacher_model
        self.student_model = student_model

        self.student_device = student_device
        self.teacher_device = teacher_device

        self.lang_model = lang_model
        self.logger = logger

        self.student_lang_model = config_json['model_kwargs']['student_lang_model']

        self.logger.log_line()
        self.logger.log_message(f'Initializing Dataloaders')
        self._init_ppo_dataloader(config_json['dataset_kwargs'])

        optimizer_kwargs = config_json['optimizer_kwargs']
        lr_scheduler_kwargs = config_json['lr_scheduler_kwargs']

        self.logger.log_line()
        self.logger.log_message(f'Initializing optimizer')
        self.logger.log_new_line()
        
        self._init_optimizer(optimizer_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_message(f'')
        self.logger.log_new_line()        

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line()

        ppo_kwargs = config_json['ppo_kwargs']
        trainer_kwargs = config_json['trainer_kwargs']

        self.teacher_mixed_sample = ppo_kwargs['teacher_mixed_sample']
        self.temperature = ppo_kwargs['temperature']
        self.reward_scaling = ppo_kwargs['reward_scaling']
        self.reward_cliprange = ppo_kwargs['reward_cliprange']
        self.seed_ppo = ppo_kwargs['seed_ppo']

        self.training_epochs = trainer_kwargs['training_epochs']
        self.ppo_epochs = trainer_kwargs['ppo_epochs']
        self.kd_ratio = trainer_kwargs['kd_ratio']
        self.gradient_accumulation_steps = trainer_kwargs["gradient_accumulation_steps"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]

        self.num_training_steps = self.total_train_batch*self.training_epochs
        self.num_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"] if lr_scheduler_kwargs["num_warmup_steps"] != -1 else self.num_training_steps//10
        self.num_warmup_steps = min(self.num_warmup_steps, lr_scheduler_kwargs["max_warmup_steps"])        

        self.logger.log_message(f'Initializing Learning Rate Scheduler')
        self._init_lr_scheduler(lr_scheduler_kwargs)      

        generation_kwargs = dict(
            do_sample=True,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            max_length=256,
            eos_token_id=self.train_ppo_loader.collate_fn.eos_token_id,
            pad_token_id=self.train_ppo_loader.collate_fn.pad_token_id,            
        )
        self.generation_config = GenerationConfig(**generation_kwargs)

        self._initialize_ppo_storage()
        self.ppo_storage.clear()

    def _init_ppo_dataloader(self, dataset_kwargs:dict):

        if dataset_kwargs['_type'] == "cnn_dailymail":
            cnn_dataset_kwargs = dataset_kwargs['cnn_dataset_kwargs']
            train_csv_path = os.path.join(cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['train_dataset'])
            eval_csv_path = os.path.join(cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['validation_dataset'])

            train_dataset = CNNDataset(csv_path=train_csv_path, dataset_type='train')
            val_dataset = CNNDataset(csv_path=eval_csv_path, dataset_type='validation')

            train_ppo_collate_fn = PPO_CNNCollatefn(
                lang_model=self.student_lang_model,
                dataset_type=train_dataset.dataset_type
            )

            self.train_ppo_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cnn_dataset_kwargs['train_batch_size'],collate_fn=train_ppo_collate_fn, shuffle=True
            )

            train_ppo_collate_fn = PPO_CNNCollatefn(
                lang_model=self.student_lang_model,
                dataset_type=val_dataset.dataset_type
            )

            self.val_ppo_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cnn_dataset_kwargs['val_batch_size'],collate_fn=train_ppo_collate_fn, shuffle=True
            )

            self.student_model.resize_token_embeddings(
                len(self.train_ppo_loader.collate_fn.tokenizer)
            )

            self.teacher_model.resize_token_embeddings(
                len(self.train_ppo_loader.collate_fn.tokenizer)
            )            

        self.total_train_batch = len(self.train_ppo_loader)
        self.ten_percent_train_batch = self.total_train_batch // 10            

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

    def _initialize_ppo_storage(self):

        self.ppo_storage = PPOStorage(
            self.train_ppo_loader.collate_fn.pad_token_id, 
            self.seed_ppo
        )

    def get_input_batch(self, ppo_data_items:dict, data_items:dict):
        
        def combine_and_left_pad(input_ids:torch.tensor, label_ids:torch.tensor, pad_token_id):
            
            total_length = input_ids.size(1) + label_ids.size(1)
            combined_sequences = torch.ones(input_ids.size(0), total_length, dtype=torch.long) * pad_token_id

            label_indices = []

            for idx, (input_seq, label_seq) in enumerate(zip(input_ids, label_ids)):
                input_seq = input_seq[input_seq!=pad_token_id]
                label_seq = label_seq[label_seq!=pad_token_id]

                combined_seq = torch.cat([input_seq, label_seq], dim=0)
                combined_seq_len = combined_seq.size(0)

                combined_sequences[idx, -combined_seq_len:] = combined_seq
                seq_len = total_length - combined_seq_len

                response_start_idx = total_length - label_seq.size(0)
                response_end_idx = total_length 

                label_indices.append([response_start_idx, response_end_idx])

            return combined_sequences, torch.tensor(label_indices)

        ''' 
        Concatenate ppo_data_items and data_items along batch dim. 
        The first 'bs' indices are ppo_batch_items the remaining len - bs = data_items. 
        '''
        query_tensors = ppo_data_items['query_tensors']
        response_tensors = ppo_data_items['response_tensors']

        # combined_ids = torch.cat([query_tensors, response_tensors], dim=1)[:, -self.max_prompt_length:]        
        combined_ids = torch.cat([query_tensors, response_tensors], dim=1)

        ppo_batch = {
            "input_ids":combined_ids,
            "attention_mask":combined_ids.not_equal(self.train_ppo_loader.collate_fn.pad_token_id).long()
        }

        data_items['input_ids'], lm_label_indices = combine_and_left_pad(
            data_items['input_ids'], data_items['label_ids'], 
            self.train_ppo_loader.collate_fn.pad_token_id
        )

        data_items['attention_mask'] = data_items['input_ids'].not_equal(
            self.train_ppo_loader.collate_fn.pad_token_id
        ).long()

        # del data_items['label_ids']

        input_batch = {}

        for k in ppo_batch:
            input_batch[k] = torch.cat([
                ppo_batch[k], data_items[k]
            ], dim=0)

        return input_batch, lm_label_indices

    def get_ppo_rl_batch(self, ppo_data_items:dict):

        return PPORLBatch(
            batch_query_tensors=ppo_data_items['query_tensors'],
            batch_response_tensors=ppo_data_items['response_tensors'],
            batch_scores=ppo_data_items['scores'],
            batch_inf_masks=ppo_data_items['inf_masks'],
            batch_rev_kl=ppo_data_items['rev_kls'],
            batch_t_reward=ppo_data_items['t_rewards'],
            batch_ent_reward=ppo_data_items['ent_rewards'],
            batch_reward=ppo_data_items['rewards'],
            batch_ids=ppo_data_items['ids'],
            batch_full_label_ids=ppo_data_items['full_label_ids'],
            batch_logprobs=ppo_data_items['logprobs'],
            batch_w=ppo_data_items['w']
        )

    def reinit_dataloader(self,train_batch_size:int):
        '''
        TODO: Fix this later, find elegant way around this!!
        '''
        train_ppo_collate_fn = PPO_CNNCollatefn(
                lang_model=self.student_lang_model,
                dataset_type=self.train_ppo_loader.dataset.dataset_type
            )

        train_ppo_loader =  torch.utils.data.DataLoader(
            self.train_ppo_loader.dataset, batch_size = train_batch_size,
            collate_fn=train_ppo_collate_fn, shuffle=True  
        )
        return train_ppo_loader
    
    def train(self):

        self.cur_epoch = 0

        # self.ppo_lm_dataloader = self.ppo_storage.create_loader(
        #                                     self.train_ppo_loader.batch_size)
        # self.ppo_lm_iterator = iter(self.ppo_lm_dataloader)

        self.max_prompt_length = self.train_ppo_loader.collate_fn.max_prompt_length

        for training_epoch in range(self.training_epochs):
            
            self.cur_epoch = training_epoch
            
            self.logger.log_line()
            self.logger.log_message(f'Current Training Epoch: {self.cur_epoch} - Evaluating Policy on Training Dataset')
            self.logger.log_new_line()
            
            self.evaluate_policy_on_train()

            self.logger.log_message(f'Training Model ')

            self.student_model.train()
            self.teacher_model.eval()

            total_loss = 0.0
            ten_per_batch_loss = 0.0
            
            total_rl_loss = 0.0
            total_lm_loss = 0.0
            
            ten_perc_rl_loss = 0.0
            ten_perc_lm_loss = 0.0
            
            reinit_train_ppo_loader = self.reinit_dataloader(4)
            
            self.ppo_lm_dataloader = self.ppo_storage.create_loader(
                                            reinit_train_ppo_loader.batch_size)
            self.ppo_lm_iterator = iter(self.ppo_lm_dataloader)
            
            for ppo_epoch in range(self.ppo_epochs):
                for batch_idx, (data_items, _) in enumerate(reinit_train_ppo_loader):
                # for batch_idx, (data_items, _) in enumerate(self.train_ppo_loader):
                
                    self.optimizer.zero_grad()

                    ppo_data_items = next(self.ppo_lm_iterator)
                    batch_size = ppo_data_items['query_tensors'].size(0)
                    input_data_items, lm_label_indices = self.get_input_batch(ppo_data_items, data_items)

                    input_data_items = self.port_to_device(input_data_items, mode='student')

                    with torch.set_grad_enabled(True):
                        outputs = self.student_model(**input_data_items)

                    logits = outputs.logits #(bs, max_inp_len + max_op_len, vocab_size)

                    ppo_logits = logits[:batch_size]
                    logits = logits[batch_size:]

                    ppo_rl_batch = self.get_ppo_rl_batch(ppo_data_items)
                    
                    del outputs

                    teacher_logits, _ = self.compute_logits_and_logprobs(
                        ppo_rl_batch.batch_query_tensors.to(self.teacher_device),
                        ppo_rl_batch.batch_response_tensors.to(self.teacher_device),
                        base='teacher', inf_mask=ppo_rl_batch.batch_inf_masks
                    )

                    lm_loss, lm_stats = Losses.compute_pt_loss(logits, teacher_logits.to(self.student_device), data_items, 
                                           self.train_ppo_loader.collate_fn.pad_token_id, lm_label_indices,
                                           self.kd_ratio)


                    rl_loss, rl_stats = Losses.compute_ppo_loss(ppo_logits=ppo_logits, teacher_logits=teacher_logits.to(self.student_device),
                                                       ppo_rl_batch=ppo_rl_batch, temperature=self.temperature,
                                                       pad_token_id=self.train_ppo_loader.collate_fn.pad_token_id,
                                                    )
                    
                    loss = rl_loss + lm_loss
                    
                    print(f'RL Loss: {rl_loss:.4f} LM Loss: {lm_loss:.4f}')
                    
                    loss = loss/self.gradient_accumulation_steps if self.gradient_accumulation_steps != 0 else loss
                    loss.backward()
                    
                    
                    if self.gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.gradient_clipping)

                    if not self.gradient_accumulation_steps:
                        self.optimizer.step()
                        self.lr_scheduler.step()

                    elif (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()

                    total_loss += loss.item()
                    ten_per_batch_loss += loss.item()
                    
                    total_rl_loss += rl_loss.item()
                    total_lm_loss += lm_loss.item()
                    
                    ten_perc_rl_loss += rl_loss.item()
                    ten_perc_lm_loss += lm_loss.item()

                    if (batch_idx + 1)%self.ten_percent_train_batch == 0:
                        average_loss = ten_per_batch_loss/self.ten_percent_train_batch
                        average_rl_loss = ten_perc_rl_loss/self.ten_percent_train_batch
                        average_lm_loss = ten_perc_lm_loss/self.ten_percent_train_batch
                        
                        perplexity = torch.exp(torch.tensor(average_loss))
                        self.logger.log_message(f'Epoch: {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - loss:{average_loss:.4f} - rl_loss: {average_rl_loss:.4f} - lm_loss: {average_lm_loss:.4f}  perplexity: {perplexity:.4f}')

                        ten_per_batch_loss = 0.0                    
                        ten_perc_rl_loss = 0.0
                        ten_perc_lm_loss = 0.0
                        
            average_loss = total_loss/len(self.train_dataloader)
            average_rl_loss = total_rl_loss/len(self.train_dataloader)
            average_lm_loss = total_lm_loss/len(self.train_dataloader)
                        
            perplexity = torch.exp(torch.tensor(average_loss))
            self.logger.log_line()
            self.logger.log_message(f'Training Epoch: {self.cur_epoch} - loss: {average_loss:.4f} - rl_loss: {average_rl_loss:.4f} - lm_loss: {average_lm_loss:.4f} perplexity: {perplexity:.4f}')
            self.logger.log_new_line()

            self.ppo_storage.clear()            

    def evaluate_policy_on_train(self):

        def get_rev_kl(log_p, log_q, mask):
            log_ratio = (log_p - log_q) * mask
            kl = log_ratio.float().exp() - 1 - log_ratio
            return kl

        self.student_model.eval()
        self.teacher_model.eval()

        ppo_rl_elements = []

        i = 0

        for data_items, meta_data_items in tqdm(self.train_ppo_loader, desc=f'Evaluating Policy on Training Dataset'):
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.student_device)
            i+=1
            query_ids = data_items['input_ids']
            _ids = meta_data_items['_ids']
            full_label_ids = meta_data_items['full_label_ids']

            del meta_data_items   

            with torch.no_grad():
                generated_outputs = self.generate_sequences(data_items)

                del data_items

                generated_ids = generated_outputs.sequences
                gen_logits = generated_outputs.scores 
                inf_masks = torch.isinf(gen_logits)

                masks = (generated_ids != self.train_ppo_loader.collate_fn.pad_token_id)
                lengths = torch.sum(masks, dim=-1)

                rewards_dict = self.compute_reward(query_ids, generated_ids, inf_masks)
                t_rewards = rewards_dict['rewards']
                inf_masks = rewards_dict['inf_masks']

                # if self.teacher_mixed_sample:
                _, rollout_logprobs = self.compute_logits_and_logprobs(query_ids, generated_ids, inf_mask=inf_masks)
                

            rev_kl = get_rev_kl(t_rewards, rollout_logprobs, masks)

            raw_logprobs = rollout_logprobs
            logprobs = rollout_logprobs
            w = torch.ones_like(logprobs)
            ent_rewards = -logprobs

            rewards = t_rewards + ent_rewards
            w = torch.ones_like(logprobs)

            if self.reward_scaling:
                rewards /= self.reward_scaling
            
            if self.reward_cliprange:
                rewards /= self.reward_cliprange

            batch_size = query_ids.size(0)
            
            ppo_rl_elements.extend(
            # self.ppo_storage.push(
                [PPORLElement(
                    query_tensor=query_ids[i].cpu(), 
                    response_tensor=generated_ids[i].cpu(), 
                    # scores=gen_logits[i].cpu(),
                    scores=None,
                    # inf_masks=inf_masks[i].cpu(),
                    inf_masks=None,
                    rev_kl=rev_kl[i].cpu(),
                    id=_ids[i],
                    full_label_id=full_label_ids[i].cpu(),
                    t_reward=t_rewards[i].cpu(),
                    ent_reward=ent_rewards[i].cpu(),
                    reward=rewards[i].cpu(),
                    logprobs=logprobs[i].cpu(),
                    w=w[i].cpu()
                )   for i in range(batch_size)]
            )

            if len(ppo_rl_elements) % 10 == 0 :
                self.ppo_storage.push(
                ppo_rl_elements
                )

                ppo_rl_elements = []
                

            gc.collect()
            if i==10:
                break
        
        if ppo_rl_elements:        
            self.ppo_storage.push(
                ppo_rl_elements
            )

        del ppo_rl_elements

    def compute_logits_and_logprobs(self, input_ids:torch.tensor, generated_ids:torch.tensor, base="student", inf_mask:torch.tensor=None):

        full_ids = torch.cat([input_ids, generated_ids], dim=-1)
        attention_mask = (full_ids != self.train_ppo_loader.collate_fn.pad_token_id)

        model_inputs = {
            "input_ids": full_ids,
            "attention_mask":attention_mask, 
            "use_cache":False
        }

        with torch.no_grad():
            if base == "student":      
                model_inputs = self.port_to_device(model_inputs, mode=base)      
                outputs = self.student_model(**model_inputs)

            elif base == "teacher":
                model_inputs = self.port_to_device(model_inputs, mode=base)
                outputs = self.teacher_model(**model_inputs)            

        logits = outputs.logits 
        logits = logits / self.temperature

        start = input_ids.size(1) - 1
        end = input_ids.size(1) + generated_ids.size(1) - 1
        mask = model_inputs['attention_mask'][:, start:end]

        logits = logits[:, start:end, :]
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf")) 
            logprobs = logprobs.masked_fill(inf_mask, -float("inf"))

        logprobs = torch.gather(logprobs, dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
        logprobs = logprobs.masked_fill(~(mask.bool()), 0)

        assert all((~torch.isinf(logprobs.view(-1))) & (~torch.isnan(logprobs.view(-1))))
        return logits, logprobs

    def compute_reward(self, input_ids:torch.tensor, generated_ids:torch.tensor, inf_mask:torch.tensor=None, output_pos:torch.tensor=None):

        full_ids = torch.cat([input_ids, generated_ids], dim=-1)
        attention_mask = (full_ids != self.train_ppo_loader.collate_fn.pad_token_id)

        model_inputs = {
            "input_ids": full_ids,
            "attention_mask":attention_mask, 
            "use_cache":False
        }

        with torch.no_grad():
            outputs = self.teacher_model(**model_inputs)
            logits = outputs.logits 
        
        logits = logits - torch.mean(logits, dim=-1, keepdim=True)

        logits = logits * attention_mask.unsqueeze(-1) # set the values corresponding to pad token to 0
        logits = logits[:, input_ids.size(-1)-1: , :]
        mask = attention_mask[:, input_ids.size(-1)-1:]

        selection_value = torch.gather(
            logits[:, :-1, :], -1, model_inputs["input_ids"][:, input_ids.size(-1):, None]
        ).squeeze(-1)

        current_logits = logits[:, :-1, :]
        next_state_value = torch.logsumexp(current_logits.float(), dim=-1)
        next_state_value = next_state_value * mask[:, :-1]

        scores = selection_value - next_state_value

        assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1))))
        assert scores.size() == generated_ids.size()

        return {
            "rewards":scores, "inf_masks":inf_mask
        }

    def generate_sequences(self, data_items):            
        max_new_tokens = self.generation_config.max_length
        generated_outputs = self.student_model.generate(
            input_ids=data_items['input_ids'],
            attention_mask=data_items['attention_mask'],
            generation_config=self.generation_config,
            max_new_tokens=max_new_tokens,
            mix_in_model=None,
            mix_in_alpha=None,
            output_scores=True,
            return_dict_in_generate=True               
        )

        generated_outputs.sequences = torch.nn.functional.pad(
            generated_outputs.sequences,
            (0, self.generation_config.max_length - generated_outputs.sequences.shape[1]),
            value=self.train_ppo_loader.collate_fn.pad_token_id,
        )
        
        if generated_outputs.scores is not None:
            generated_outputs.scores = torch.stack(generated_outputs.scores, dim=1)
            
            # additional_zeros = self.generation_config.max_length - self.train_ppo_loader.collate_fn.max_prompt_length - generated_outputs.scores.size(1)
            additional_zeros = generated_outputs.sequences.size(1) - generated_outputs.scores.size(1)
            if additional_zeros > 0:
                generated_outputs.scores = torch.cat([
                    generated_outputs.scores, 
                    torch.zeros(
                        generated_outputs.scores.size(0),
                        additional_zeros,
                        generated_outputs.scores.size(2),
                        device=generated_outputs.scores.device)],
                    dim=1)
            
        return generated_outputs
    
    def port_to_device(self, data_items:dict, mode:str="student"):
        if mode == "student":
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.student_device)    

        elif mode == "teacher":
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.teacher_device)    

        return data_items