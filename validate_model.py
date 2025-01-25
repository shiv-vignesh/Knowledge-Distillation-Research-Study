import os , json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset_utils.cnn_dataset import CNNCollatefn, CNNDataset
from dataset_utils.utils import top_k_top_p_filtering

from rouge_score import rouge_scorer

from tqdm import tqdm

def load_model(model_name:str, ckpt_path:str, device_id:int, tokenizer:GPT2Tokenizer):

    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else "cpu"

    model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    if os.path.exists(ckpt_path):
        print('Loading saved model ckpt')
        state_dict = torch.load(ckpt_path, map_location=torch.device(device))
        '''Saved using dataparallel. state_dict key mismatch'''
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    model.to(device)
    return model

def load_dataloader(file_path:str, dataset_type:str, batch_size:int=4):

    dataset = CNNDataset(
        file_path, dataset_type
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=CNNCollatefn(
        'gpt2', dataset_type=dataset_type, return_meta_data=True
    ))

    return dataloader

def compute_rogue_metrics(generated_highlights, reference_highlights):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    batch_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0} 
    scores_list = []
            
    for gen, ref in zip(generated_highlights, reference_highlights):
        score = scorer.score(ref, gen)
        scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0} 
        for key in scores.keys():
            batch_scores[key] += score[key].fmeasure
            scores[key] = score[key].fmeasure

        scores_list.append(scores)

    # Average the scores
    for key in scores.keys():
        batch_scores[key] /= len(generated_highlights)

    return batch_scores, scores_list                

def generate_batch_text(model:GPT2LMHeadModel, data_items: dict, device:torch.device, max_generation_length:int, pad_token_id, model_max_length):
    input_ids = data_items["inference_ids"].to(device)

    batch_size, current_max_length = input_ids.size()
    max_length = current_max_length + max_generation_length

    new_input_ids = torch.full((batch_size, max_length), fill_value=pad_token_id, device=device)

    non_pad_mask = input_ids != pad_token_id
    last_non_pad_indices = non_pad_mask.sum(dim=1) - 1 # [indices: size = bs]

    for step in range(max_generation_length):
        with torch.no_grad():
            outputs = model(input_ids=input_ids).logits

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
            next_token_logit = outputs[idx, last_non_pad_idx, :]/ 0.7
            # next_token_logit = outputs[idx, last_non_pad_idx, :-1]/ 0.8  # Assuming temperature of 1.0; -1 for avoiding PAD token which is last idx in vocab
            filtered_logit = top_k_top_p_filtering(next_token_logit)

            next_token = torch.multinomial(F.softmax(filtered_logit, dim=-1), num_samples=1)
            # next_step_input_ids[idx] = next_token

            new_input_ids[idx][:last_non_pad_idx + 1] = input_ids[idx][:last_non_pad_idx+1].clone() 
            new_input_ids[idx][last_non_pad_idx+1] = next_token

        if new_input_ids.size(-1) < model_max_length:
            input_ids = new_input_ids    
        else:                
            input_ids = new_input_ids[:, max_generation_length:]
        
    return input_ids, last_non_pad_indices

def validate(model:GPT2LMHeadModel, dataloader:DataLoader, save_path:str):

    device = model.device

    model.eval()

    validation_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0} 
    total_batches = 0

    prediction_json = []

    iterator = tqdm(dataloader, desc=f'Running Inference')

    for idx, data_items in enumerate(iterator):
        for k,v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(model.device)

        batch_highlights = data_items["highlights"] if "highlights" in data_items else None
        batch_articles = data_items["articles"] if "articles" in data_items else None 
        batch_article_lengths = data_items["batch_article_lengths"] if "batch_article_lengths" in data_items else None 
        batch_sentiment_label = data_items["batch_sentiment_label"] if "batch_sentiment_label" in data_items else None 
        batch_length_bucket = data_items["batch_length_bucket"] if "batch_length_bucket" in data_items else None        

        input_ids, last_non_pad_indices = generate_batch_text(
            model=model, data_items=data_items, device=device,
            max_generation_length=100, pad_token_id=dataloader.collate_fn.tokenizer.pad_token_id,
            model_max_length=dataloader.collate_fn.tokenizer.model_max_length
        )

        batch_generated_highlights = []
        
        for batch_idx, input_id in enumerate(input_ids):
            last_non_pad_idx = last_non_pad_indices[batch_idx]
            generated_ids = input_id[last_non_pad_idx:].tolist() 
            text = dataloader.collate_fn.tokenizer.decode(generated_ids, skip_special_tokens=True)

            batch_generated_highlights.append(text)

        scores, scores_list = compute_rogue_metrics(batch_generated_highlights, batch_highlights)

        for key in validation_scores.keys():
            validation_scores[key] += scores[key]        

        batch_predictions = []
        for article, target, generated, score, sentiment_label, length_bucket in zip(batch_articles, batch_highlights, batch_generated_highlights, scores_list, batch_sentiment_label, batch_length_bucket):
            batch_predictions.append({
                "article":article, "target_highlight": target, "generated_highlight":generated, "rouge":score, "sentiment":sentiment_label, "length_bucket":length_bucket, "category": ""
            })

        prediction_json.extend(batch_predictions)

        with open(f'{save_path}','w+') as f:
            json.dump(prediction_json, f)        

    for key in validation_scores.keys():
        validation_scores[key] /= len(dataloader)  

    return validation_scores

if __name__ == "__main__":

    csv_path = "data/final/subset_test.csv"
    model_path = 'model_ckpts/20Apr24_KD_JS_DIV/model_checkpoints/n_batch_ckpt.pt'
    device_id = 9
    model_name = 'gpt2'

    dataloder = load_dataloader(
        csv_path, dataset_type='validation', batch_size=8
    )

    model = load_model(
        model_name, model_path, device_id=device_id, tokenizer=dataloder.collate_fn.tokenizer
    )

    output_dir = "final_prediction_samples/20Apr24_KD_JS_DIV"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scores = validate(
        model, dataloder, f'{output_dir}/subset_test_prediction.json'
    )

    with open(f'{output_dir}/validation_test_scores.json','w+') as f:
        json.dump(scores, f)