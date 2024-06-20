import pandas as pd

import torch 
from torch.utils.data import Dataset

from transformers import GPT2Tokenizer

class CNNDataset(Dataset):
    def __init__(self, csv_path:str, dataset_type:str):
        self.data = pd.read_csv(csv_path)
        self.dataset_type = dataset_type

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row.to_dict()
    
    def __len__(self):
        return len(self.data)

class CNNCollatefn(object):
    def __init__(self, lang_model:str, dataset_type:str, special_token:str=None, truncation:bool=True, padding:str="longest", return_meta_data:bool=False):

        if "gpt2" in lang_model:
            self.tokenizer = GPT2Tokenizer.from_pretrained(lang_model)

        self.special_token = special_token
        self.truncation = truncation
        self.padding = padding

        self.dataset_type = dataset_type

        self.start_token = "<|startoftext|>"
        self.summarize_token = "summarize:"
        self.eos_token = "<|endoftext|>"
        self.sep_token = "</s>"

        self.highlight_start_token = "<highlight_start>"
        self.highlight_end_token = "<highlight_end>"

        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        
        # self.tokenizer.add_special_tokens({'pad_token': self.pad_token, 'mask_token':self.mask_token})
        self.tokenizer.add_special_tokens({'pad_token': self.pad_token})
        self.return_meta_data = return_meta_data


    def collate_fn(self, data_items:dict):
        
        batch_input_texts = []
        batch_inference_texts = []
        batch_highlights = []

        batch_articles = []
        batch_article_lengths = []
        batch_length_bucket = []
        batch_sentiment_label = []
        batch_ids = []

        for data_item in data_items:
            _id = data_item["id"]

            article = data_item["article"]
            highlights = data_item["highlights"]

            article_length = data_item["article_length"] if "article_length" in data_item else None
            length_bucket = data_item["length_bucket"] if "length_bucket" in data_item else None
            sentiment_label = data_item["sentiment_label"] if "sentiment_label" in data_item else None

            '''<|startoftext|> summarize: ---- </s>'''
            article_prompt = f'{self.start_token} {self.summarize_token} {article} {self.sep_token}'
            
            if self.dataset_type == "train":
                '''<|startoftext|> summarize: ---- </s> highlights <|endoftext|>'''
                summary_prompt = f'{self.highlight_start_token} {highlights} {self.highlight_end_token}'
                input_text = article_prompt + ' ' + summary_prompt
                batch_input_texts.append(input_text)

            else:
                '''<|startoftext|> summarize: ---- </s>'''
                inference_text = article_prompt
                eval_text = article_prompt + f'{self.highlight_start_token} {highlights} {self.highlight_end_token}'
                
                batch_input_texts.append(eval_text)
                batch_inference_texts.append(inference_text)

                batch_highlights.append(highlights)     
                batch_articles.append(article)     

            if self.return_meta_data:
                batch_article_lengths.append(article_length)
                batch_sentiment_label.append(sentiment_label)
                batch_length_bucket.append(length_bucket)  
            
            batch_ids.append(_id)
        
        input_ids = self.tokenizer(batch_input_texts, return_tensors="pt", truncation=self.truncation, padding=self.padding)

        if self.dataset_type != "train":
            inference_input_ids = self.tokenizer(batch_inference_texts, return_tensors="pt", truncation=self.truncation, padding=self.padding)

        if self.dataset_type == "train":
            # print(f'From Collate_fn() {input_ids["input_ids"]}')
            
            return {
                "input_ids":input_ids['input_ids'],
                "attention_mask":input_ids['attention_mask'],
                "labels":input_ids['input_ids'],
                "_ids":batch_ids
            }
        
        else:
            if self.return_meta_data:
                return {
                    "input_ids":input_ids['input_ids'],
                    "attention_mask":input_ids['attention_mask'],
                    "labels":input_ids['input_ids'],
                    "highlights":batch_highlights,
                    "articles":batch_articles,
                    "inference_ids":inference_input_ids["input_ids"],
                    "batch_article_lengths":batch_article_lengths, 
                    "batch_sentiment_label":batch_sentiment_label, 
                    "batch_length_bucket":batch_length_bucket,
                    "_ids":batch_ids
                }
            else:
                return {
                    "input_ids":input_ids['input_ids'],
                    "attention_mask":input_ids['attention_mask'],
                    "labels":input_ids['input_ids'],
                    "highlights":batch_highlights,
                    "articles":batch_articles,
                    "inference_ids":inference_input_ids["input_ids"],
                    "_ids":batch_ids
                }                


    def __call__(self, data_items):
        return self.collate_fn(data_items)
        