import os, json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import get_gpu_count, check_gpu_availability, create_logger, top_k_top_p_filtering, filter_top_percent, save_tensor2pickle, save_tensor2npy
from dataset_utils.cnn_dataset import CNNDataset, CNNCollatefn

# import deepspeed

class LogitsGenerator:

    def __init__(self, model_arch:dict, 
                 dataset_kwargs:dict, 
                 processing_strategy_kwargs:dict,
                 output_kwargs:dict):

        self.logger = create_logger(output_kwargs['output_dir'])
        self.teacher_model_name = model_arch['teacher_lang_model']  

        self.logger.log_line()
        self.logger.log_message(f'Initializing Dataloaders')
        self._init_dataloader(dataset_kwargs)

        self.logger.log_message(f'Creating Teacher Model: {self.teacher_model_name} Object')
        self.load_model(model_arch)
        self.logger.log_message(f'Model Name - {self.teacher_model_name} GPU - {self.teacher_device}')

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

        self.temperature = processing_strategy_kwargs['temperature']
        self.logits_processing_strategy = processing_strategy_kwargs['logits_processing_strategy']
        self.nucleus_sampling = processing_strategy_kwargs["nucleus_sampling"]
        self.percent_k = processing_strategy_kwargs['percent_k']
        self.save_format = processing_strategy_kwargs['save_format']

        self.output_dir = output_kwargs['output_dir']

        self.logger.log_line()
        self.logger.log_message(f'Processing Strategy Kwargs')
        self.logger.log_line()
        self.logger.log_message(f'Temperature Scaling Value - {self.temperature}')
        self.logger.log_message(f'logits processing strategy - {self.logits_processing_strategy}')
        self.logger.log_message(f'Nucleus Sampling - {self.nucleus_sampling}')
        self.logger.log_message(f'Top k Percent - {self.percent_k}')
        self.logger.log_message(f'Logits Save Format - {self.save_format}')
        self.logger.log_new_line()

    def load_model(self, model_arch:dict):

        if "gpt2" in self.teacher_model_name.lower():
            self.teacher_model = GPT2LMHeadModel.from_pretrained(self.teacher_model_name)
            self.port_model2device(model_arch)

            self.logger.log_message(f'Resizing {self.teacher_model_name}')
            
            self.teacher_model.resize_token_embeddings(
                len(self.train_dataloader.collate_fn.tokenizer)
            )

            if os.path.exists(model_arch['teacher_model_ckpt_path']):
                self.load_model_ckpt(model_arch)
            else:
                self.logger.log_message(f'Unable to find Model Ckpt - {model_arch["teacher_model_ckpt_path"]}')
                self.logger.log_new_line()

    def load_model_ckpt(self, model_arch:dict):
        state_dict = torch.load(model_arch['teacher_model_ckpt_path'], map_location=torch.device(self.teacher_device))
        '''Saved using dataparallel. state_dict key mismatch'''
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if new_state_dict:
            self.teacher_model.load_state_dict(new_state_dict)
        else:
            self.teacher_model.load_state_dict(state_dict)


    def port_model2device(self, model_arch:dict):
        if check_gpu_availability():
            gpu_count = get_gpu_count()

            if gpu_count > 1 and model_arch['gpu_id'] <= gpu_count:
                self.teacher_device_id = model_arch['gpu_id']
                self.teacher_device = torch.device(f"cuda:{self.teacher_device_id}")
            else:
                self.teacher_device_id = 0
                self.teacher_device = torch.device("cuda")

            self.teacher_model.to(self.teacher_device)

        else:
            self.teacher_device_id = "cpu"
            self.teacher_device = "cpu"

    # def port_model2dsengine(self):
        
    #     '''TODO, initialize config for deepspeed'''
    #     self.teacher_model = deepspeed.init_inference(
    #         model=self.teacher_model,

    #     )        

    def _init_dataloader(self, dataset_kwargs:dict, multi_gpu:bool=False):        
        def _init_cnn_dataloader_helper(root_dir:str, csv_file:str, batch_size:int, dataset_type:str):
            dataset = CNNDataset(
                f'{root_dir}/{csv_file}',dataset_type
            )

            if not multi_gpu:
                dataloader = DataLoader(
                    dataset, batch_size=batch_size,
                    collate_fn=CNNCollatefn(
                        self.teacher_model_name,
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
                        self.teacher_model_name,
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

            self.train_dataloader = None 
            self.validation_dataloader = None
            
            if os.path.exists(cnn_dataset_kwargs['data_dir']):
                
                if cnn_dataset_kwargs['train_dataset'] or cnn_dataset_kwargs['train_dataset'] is not None:
                    self.train_dataloader = _init_cnn_dataloader_helper(
                        cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['train_dataset'],
                        cnn_dataset_kwargs['train_batch_size'], dataset_type='train'
                    )

                if cnn_dataset_kwargs['validation_dataset'] or cnn_dataset_kwargs['validation_dataset'] is not None:
                    self.validation_dataloader = _init_cnn_dataloader_helper(
                        cnn_dataset_kwargs['data_dir'], cnn_dataset_kwargs['validation_dataset'],
                        cnn_dataset_kwargs['val_batch_size'], dataset_type='validation'                
                    )

            else:
                self.logger.log_message(f'Unable to locate data directory: {cnn_dataset_kwargs["data_dir"]}. Exiting Process!')
                exit(1)

            if self.train_dataloader is None and self.validation_dataloader is None:
                self.logger.log_message(f'Both Training and Validation Datasets are missing. Exiting Process!')
                exit(1)

    def filter_generated_logits(self, teacher_logits:torch.tensor, data_items:dict):

        sep_token_id = self.train_dataloader.collate_fn.tokenizer.convert_tokens_to_ids("</s>")        
        pad_token_id = self.train_dataloader.collate_fn.tokenizer.pad_token_id
        highlight_start_id = self.train_dataloader.collate_fn.tokenizer.convert_tokens_to_ids("<highlight_start>")
        highlight_end_id = self.train_dataloader.collate_fn.tokenizer.convert_tokens_to_ids("<highlight_end>")

        labels = data_items['input_ids']
        highlight_start_index = (labels == highlight_start_id).nonzero(as_tuple=True)[1]
        highlight_end_index = (labels == highlight_end_id).nonzero(as_tuple=True)[1]
        
        batch_filtered_logits = []

        for batch_idx, logits in enumerate(teacher_logits):
            start_idx, end_idx = highlight_start_index[batch_idx], highlight_end_index[batch_idx]
            batch_filtered_logits.append(
                logits[start_idx:end_idx, :]
            )        

        #     print(logits[start_idx:end_idx, :].size())
        # exit(1)

        return batch_filtered_logits

    def generate(self):
        
        self.teacher_model.eval()
        iterator = tqdm(self.train_dataloader, desc=f'Generating Train Logits with batch_size - {self.train_dataloader.batch_size}')

        for batch_idx, data_items in enumerate(iterator):
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.teacher_model.device)
            
            with torch.no_grad():
                teacher_logits = self.teacher_model(data_items['input_ids']).logits

            #apply temperature
            teacher_logits /= self.temperature            
            
            if self.nucleus_sampling:
                teacher_logits = top_k_top_p_filtering(teacher_logits)

            if self.percent_k > 0.00 or self.percent_k == -1:
                teacher_logits = filter_top_percent(teacher_logits, self.percent_k)

            teacher_logits = self.filter_generated_logits(teacher_logits, data_items)

            if not os.path.exists(f'{self.output_dir}/logits'):
                os.makedirs(f'{self.output_dir}/logits')

            save_dir = f'{self.output_dir}/logits'
            for idx, logits in enumerate(teacher_logits):
                _id = data_items['_ids'][idx]
                
                if self.save_format == "npy":
                    save_tensor2npy(logits, f'{save_dir}/{_id}.npy')

                elif self.save_format == "pkl":
                    save_tensor2pickle(logits, f'{save_dir}/{_id}.pkl')


if __name__ == "__main__":

    config_json = json.load(open(f'configs/config_generate_teacher_logits.json'))

    logits_generator = LogitsGenerator(
        config_json['model_arch'],
        config_json['dataset_kwargs'],
        config_json['processing_strategy_kwargs'],
        config_json['output_kwargs']
    )

    logits_generator.generate()